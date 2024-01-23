import importlib
import itertools
import math
import os
import platform

import accelerate
import diffusers
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from accelerate.utils.dataclasses import LoggerType
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from diffusers.training_utils import compute_snr
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

from src.core.dataset import DreamBoothDataset
from src.core.trainer import BaseTrainer
from src.dreambooth.schemas.training import DreamboothTrainingSchema
from src.utils.common import collate_fn


class DreamboothTrainer(BaseTrainer):
    def __init__(self, schema: DreamboothTrainingSchema) -> None:
        super().__init__(schema=schema)
        self.logger.info("Initializing Dreambooth trainer...")
        self.tokenizer = self._init_tokenizer()

        # text encoder
        self.text_encoder_model_class = self._get_text_encoder_model_class()
        self.text_encoder = self._init_text_encoder()

        # register hooks for saving and loading
        self.accelerator.register_save_state_pre_hook(self._save_model_hook)
        self.accelerator.register_load_state_pre_hook(self._load_model_hook)

    def _init_tokenizer(self, sub_folder: str = "tokenizer"):
        self.logger.info(f"Initializing tokenizer from pretrained model: {self.schema.pretrained_model_name_or_path}")
        return (
            AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=self.schema.tokenizer_name,
                use_fast=False,
            )
            if self.schema.tokenizer_name
            else AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=self.schema.pretrained_model_name_or_path,
                use_fast=False,
                subfolder=sub_folder,
            )
        )

    def _get_text_encoder_model_class(self, sub_folder: str = "text_encoder"):
        text_encoder_config = PretrainedConfig.from_pretrained(
            pretrained_model_name_or_path=self.schema.pretrained_model_name_or_path,
            subfolder=sub_folder,
        )
        model_class_name = text_encoder_config.architectures[0]
        if model_class_name == "CLIPTextModel":
            from transformers import CLIPTextModel

            model_class = CLIPTextModel
        elif model_class_name == "T5EncoderModel":
            from transformers import T5EncoderModel

            model_class = T5EncoderModel
        else:
            self.logger.error(f"Unsupported Text Encoder model class: {model_class_name}")
        return model_class

    def _init_text_encoder(self, sub_folder: str = "text_encoder"):
        self.logger.info(
            f"Initializing text encoder: {self.text_encoder_model_class} from {self.schema.pretrained_model_name_or_path}"
        )
        text_encoder = self.text_encoder_model_class.from_pretrained(
            pretrained_model_name_or_path=self.schema.pretrained_model_name_or_path,
            variant=self.schema.variant,
            subfolder=sub_folder,
        )
        if self.schema.train_text_encoder and text_encoder.dtype != torch.float32:
            raise ValueError(
                f"Text encoder loaded as datatype {text_encoder.dtype}. Please make sure to always have all model weights in full float32 precision."
            )
        if not self.schema.train_text_encoder:
            text_encoder.requires_grad_(False)
        return text_encoder

    def _save_model_hook(self, models, weights, output_dir):
        if self.accelerator.is_main_process:
            for model in models:
                sub_dir = (
                    "unet" if isinstance(model, type(self.accelerator.unwrap_model(self.unet))) else "text_encoder"
                )
                model.save_pretrained(output_dir / sub_dir)
                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

    def _load_model_hook(self, models, input_dir):
        while len(models) > 0:
            # pop models so that they are not loaded again
            model = models.pop()

            if isinstance(model, type(self.accelerator.unwrap_model(self.text_encoder))):
                # load transformers style into model
                load_model = self.text_encoder_model_class.from_pretrained(
                    pretrained_model_name_or_path=input_dir, subfolder="text_encoder"
                )
                model.config = load_model.config
            else:
                load_model = UNet2DConditionModel.from_pretrained(
                    pretrained_model_name_or_path=input_dir, subfolder="unet"
                )
                model.register_to_config(**load_model.config)

            model.load_state_dict(load_model.state_dict())
            del load_model

    def _tokenize_prompt(self, prompt: str, add_special_tokens: bool = False):
        return self.tokenizer(
            text=prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            add_special_tokens=add_special_tokens,
            return_tensors="pt",
        )

    def _encode_prompt(self, text_encoder, input_ids, attention_mask, text_encoder_use_attention_mask=None):
        text_inputs_ids = input_ids.to(device=self.text_encoder.device)

        if text_encoder_use_attention_mask:
            text_attention_mask = attention_mask.to(device=self.text_encoder.device)
        else:
            text_attention_mask = None
        prompt_embeds = text_encoder(
            input_ids=text_inputs_ids,
            attention_mask=text_attention_mask,
            return_dict=False,
        )
        prompt_embeds = prompt_embeds[0]
        return prompt_embeds

    def setup(self):
        if self.schema.gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()
            if self.schema.train_text_encoder:
                self.text_encoder.gradient_checkpointing_enable()

        if self.schema.scale_learning_rate:
            self.schema.learning_rate = (
                self.schema.learning_rate
                * self.schema.gradient_accumulation_steps
                * self.schema.train_batch_size
                * self.accelerator.num_processes
            )

        parameters_to_optimize = itertools.chain(
            self.unet.parameters(),
            self.text_encoder.parameters() if self.schema.train_text_encoder else self.unet.parameters(),
        )

        optimizer = self._init_optimizer(parameter_to_optimize=parameters_to_optimize)

        # TODO: add pre_compute_text_embeddings

        train_dataset = DreamBoothDataset(
            height=self.schema.height,
            width=self.schema.width,
            tokenizer=self.tokenizer,
            instance_data_dir=self.schema.instance_data_dir,
            instance_prompt=self.schema.instance_prompt,
            num_class_images=self.schema.num_class_images,
            tokenize_prompt=self._tokenize_prompt,
            class_data_dir=self.schema.class_data_dir,
            class_prompt=self.schema.class_prompt,
            encoder_hidden_states=None,
            class_prompt_encoder_hidden_states=None,
        )

        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=self.schema.train_batch_size,
            shuffle=True,
            collate_fn=lambda examples: collate_fn(examples, self.schema.with_prior_preservation),
            num_workers=self.schema.dataloader_num_workers,
        )

        # Scheduler and math around the number of training steps.
        overrode_max_train_steps = False
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / self.schema.gradient_accumulation_steps)
        if not self.schema.max_train_steps:
            self.schema.max_train_steps = self.schema.num_train_epochs * num_update_steps_per_epoch
            overrode_max_train_steps = True

        learning_rate_scheduler = self._init_learning_rate_scheduler(optimizer=optimizer)

        # Prepare everything with accelerator.
        if self.schema.train_text_encoder:
            (
                self.unet,
                self.text_encoder,
                optimizer,
                train_dataloader,
                learning_rate_scheduler,
            ) = self.accelerator.prepare(
                self.unet,
                self.text_encoder,
                optimizer,
                train_dataloader,
                learning_rate_scheduler,
            )
        else:
            (
                self.unet,
                optimizer,
                train_dataloader,
                learning_rate_scheduler,
            ) = self.accelerator.prepare(self.unet, optimizer, train_dataloader, learning_rate_scheduler)

        if not self.schema.train_text_encoder and self.text_encoder:
            self.text_encoder.to(device=self.accelerator.device, dtype=self.weight_dtype)

        # we need to recalculate the total training steps as the size of the training dataloader may have changed
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / self.schema.gradient_accumulation_steps)
        if overrode_max_train_steps:
            self.schema.max_train_steps = self.schema.num_train_epochs * num_update_steps_per_epoch

        # afterwards we recalculate the number of training epochs
        self.schema.num_train_epochs = math.ceil(self.schema.max_train_steps / num_update_steps_per_epoch)

        # We need to initialize the trackers we use, and also store our configuration.
        # The trackers initializes automatically on the main process.
        if self.accelerator.is_main_process:
            self.accelerator.init_trackers(
                project_name=self.schema.output_dir.name, config=self.schema.model_dump(mode="json")
            )

        return train_dataloader, optimizer, learning_rate_scheduler, num_update_steps_per_epoch

    def _snr_gamma_weighting(self, model_pred, target, timesteps):
        # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
        # Since we predict the noise instead of x_0, the original formulation is slightly changed.
        # This is discussed in Section 4.2 of the same paper.
        if self.schema.with_prior_preservation:
            # if we're using prior preservation, we calc snr for instance loss only -
            # and hence only need timesteps corresponding to instance images
            snr_timesteps, _ = torch.chunk(timesteps, 2, dim=0)
        else:
            snr_timesteps = timesteps

        # compute the SNR for each timestep
        snr = compute_snr(noise_scheduler=self.noise_scheduler, timesteps=snr_timesteps)
        base_weight = (
            torch.stack([snr, self.schema.snr_gamma * torch.ones_like(snr_timesteps)], dim=1).min(dim=1)[0] / snr
        )

        if self.noise_scheduler.config.prediction_type == "epsilon":
            # Velocity objective needs to be floored to an SNR weight of one.
            mse_loss_weights = base_weight + 1
        else:
            # Epsilon and sample both use the same loss weights.
            mse_loss_weights = base_weight

        loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
        loss = loss.mean(dim=list(range(1, loss.shape))) * mse_loss_weights
        return loss.mean()

    def _log_training_details(self, train_dataloader, optimizer, learning_rate_scheduler, total_batch_size):
        self.logger.info("***** Environment and Model Architecture *****")
        self.logger.info(f"Python Version: {platform.python_version()}")
        self.logger.info(f"PyTorch Version: {torch.__version__}")
        self.logger.info(f"Diffusers Version: {diffusers.__version__}")
        self.logger.info(f"Accelerate Version: {accelerate.__version__}")

        # start of the training process
        self.logger.info("***** Training Configuration *****")
        self.logger.info(f"Mix precision: {self.schema.mixed_precision}")
        self.logger.info(f"Num examples = {len(train_dataloader.dataset)}")
        self.logger.info(f"Num batches each epoch = {len(train_dataloader)}")
        self.logger.info(f"Num Epochs = {self.schema.num_train_epochs}")
        self.logger.info(f"Instantaneous batch size per device = {self.schema.train_batch_size}")
        self.logger.info(f"Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        self.logger.info(f"Gradient Accumulation steps = {self.schema.gradient_accumulation_steps}")
        self.logger.info(f"Total optimization steps = {self.schema.max_train_steps}")

        self.logger.info("***** Device and Distributed Training Details *****")
        self.logger.info(f"Device: {self.accelerator.device}")
        self.logger.info(f"Number of GPUs: {self.accelerator.num_processes}")
        self.logger.info(f"Is distributed training: {self.accelerator.num_processes > 1}")

        self.logger.info("***** Optimizer and Learning Rate Scheduler *****")
        self.logger.info(f"Optimizer: {type(optimizer).__name__}")
        self.logger.info(f"Learning Rate: {self.schema.learning_rate}")
        self.logger.info(f"Learning Rate Scheduler: {type(learning_rate_scheduler).__name__}")

        self.logger.info("***** Checkpointing and Validation *****")
        self.logger.info(f"Checkpoint Frequency: {self.schema.checkpointing_steps} steps")
        self.logger.info(f"Validation Steps: {self.schema.validation_steps}")

        self.logger.info("***** Instance and Class Data Details *****")
        self.logger.info(f"Instance Data Directory: {self.schema.instance_data_dir}")
        self.logger.info(f"Number of Instance Images: {len(os.listdir(self.schema.instance_data_dir))}")
        self.logger.info(f"Instance Prompt: {self.schema.instance_prompt}")

        if self.schema.class_data_dir:
            self.logger.info(f"Class Data Directory: {self.schema.class_data_dir}")
            self.logger.info(f"Number of Class Images: {len(os.listdir(self.schema.class_data_dir))}")
            self.logger.info(f"Class Prompt: {self.schema.class_prompt}")
            self.logger.info(f"Prior Preservation Enabled: {self.schema.with_prior_preservation}")
            self.logger.info(f"Prior Preservation Loss Weight: {self.schema.prior_loss_weight}")

    def _validate(self, global_step: int):
        self.logger.info("***** Running Validation *****")
        self.logger.info(f"Generating {self.schema.num_validation_images} images for validation")
        self.vae = self.vae.to(device=self.vae_dtype)
        pipeline = StableDiffusionPipeline.from_pretrained(
            pretrained_model_name_or_path=self.schema.pretrained_model_name_or_path,
            tokenizer=self.tokenizer,
            text_encoder=self.text_encoder,
            unet=self.unet,
            variant=self.schema.variant,
            torch_dtype=self.weight_dtype,
            safety_checker=None,
            vae=self.vae,
        )

        scheduler_args = self._get_scheduler_args(pipeline=pipeline)

        module = importlib.import_module("diffusers")
        scheduler_class = getattr(module, self.schema.validation_scheduler.value)
        pipeline.scheduler = scheduler_class.from_config(pipeline.scheduler.config, **scheduler_args)
        pipeline = pipeline.to(device=self.accelerator.device)
        pipeline.set_progress_bar_config(disable=True)

        # TODO: add pre-computed text embeddings
        pipeline_args = {
            "prompt": self.schema.validation_prompt,
            "negative_prompt": self.schema.validation_negative_prompt,
        }

        generator = (
            torch.Generator(device=self.accelerator.device).manual_seed(seed=self.schema.seed)
            if self.schema.seed
            else None
        )

        images = []
        for _ in tqdm(range(self.schema.num_validation_images)):
            result = pipeline(
                **pipeline_args,
                generator=generator,
                num_inference_steps=self.schema.validation_num_inference_steps,
                guidance_scale=self.schema.validation_guidance_scale,
            )
            images.append(result.images[0])

        for tracker in self.accelerator.trackers:
            if tracker.name == LoggerType.TENSORBOARD.value:
                np_images = np.stack([np.asarray(image) for image in images])
                tracker.writer.add_images(
                    "validation",
                    np_images,
                    global_step=global_step,
                    dataformats="NHWC",
                )
            if tracker.name == LoggerType.WANDB.value:
                tracker.log(
                    {
                        "validation": [
                            wandb.Image(image, caption=f"{i}: {self.schema.validation_prompt}")
                            for i, image in enumerate(images)
                        ]
                    }
                )
        del pipeline
        torch.cuda.empty_cache()
        return images

    def _save_trained_model(self):
        if self.accelerator.is_main_process:
            pipeline = StableDiffusionPipeline.from_pretrained(
                pretrained_model_name_or_path=self.schema.pretrained_model_name_or_path,
                unet=self.accelerator.unwrap_model(self.unet),
                variant=self.schema.variant,
                tokenizer=self.accelerator.unwrap_model(self.tokenizer),
            )
            scheduler_args = self._get_scheduler_args(pipeline=pipeline)
            pipeline.scheduler = pipeline.scheduler.from_config(pipeline.scheduler.config, **scheduler_args)
            pipeline.save_pretrained(save_directory=self.schema.output_dir)
            self.logger.info(f"Saving trained model to {self.schema.output_dir}")
            return pipeline

    def train(self):
        train_dataloader, optimizer, learning_rate_scheduler, num_update_steps_per_epoch = self.setup()
        total_batch_size = (
            self.schema.train_batch_size * self.accelerator.num_processes * self.schema.gradient_accumulation_steps
        )
        self._log_training_details(
            train_dataloader=train_dataloader,
            optimizer=optimizer,
            learning_rate_scheduler=learning_rate_scheduler,
            total_batch_size=total_batch_size,
        )

        # Cache Latents
        if self.schema.cache_latents:
            cached_latents = self._cache_latents(train_dataloader=train_dataloader)

        # Check if continuing training from a checkpoint
        if self.schema.resume_from_checkpoint:
            global_step, initial_global_step, first_epoch = self._resume_from_checkpoint(num_update_steps_per_epoch)
        else:
            global_step = 0
            initial_global_step = 0
            first_epoch = 0

        progress_bar = tqdm(
            range(self.schema.max_train_steps),
            initial=initial_global_step,
            desc="Steps",
            # Only show the progress bar once on each machine.
            disable=not self.accelerator.is_local_main_process,
        )

        for epoch in range(first_epoch, self.schema.num_train_epochs):
            self.logger.info(f"***** Epoch {epoch} *****")
            self.unet.train()
            if self.schema.train_text_encoder:
                self.text_encoder.train()
            epoch_iterator = tqdm(
                train_dataloader,
                desc="Iteration",
                disable=not self.accelerator.is_local_main_process,
            )
            for step, batch in enumerate(epoch_iterator):
                with self.accelerator.accumulate(self.unet):
                    if self.schema.cache_latents:
                        # Convert images to latent space
                        model_input = cached_latents[step].sample()
                    else:
                        pixel_values = batch["pixel_values"].to(dtype=self.weight_dtype)
                        model_input = self.vae.encode(pixel_values).latent_dist.sample()
                    model_input = model_input * self.vae_scaling_factor

                    # sample noise for the diffusion process
                    noise = (
                        torch.randn_like(model_input)
                        + 0.1
                        * torch.randn(
                            model_input.shape[0],
                            model_input.shape[1],
                            1,
                            1,
                            device=model_input.device,
                        )
                        if self.schema.offset_noise
                        else torch.randn_like(model_input)
                    )
                    bsz, channels, height, width = model_input.shape

                    # sample a random timestep for each image in the batch
                    timesteps = torch.randint(
                        0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device
                    )

                    timesteps = timesteps.long()

                    # Forward diffusion process
                    # Add noise to the model input according to the noise magnitude at each timestep
                    noisy_model_input = self.noise_scheduler.add_noise(model_input, noise, timesteps)

                    # Get the text embeddings for conditioning
                    # TODO: add pre-computed text embeddings
                    encoder_hidden_states = self._encode_prompt(
                        text_encoder=self.text_encoder,
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        text_encoder_use_attention_mask=self.schema.text_encoder_use_attention_mask,
                    )

                    if self.accelerator.unwrap_model(self.unet).config.in_channels == channels * 2:
                        noisy_model_input = torch.cat([noisy_model_input, noisy_model_input], dim=1)

                    if self.schema.class_labels_conditioning == "timesteps":
                        class_labels = timesteps
                    else:
                        class_labels = None

                    # Predict the noise residual
                    model_pred = self.unet(
                        noisy_model_input,
                        timesteps,
                        encoder_hidden_states=encoder_hidden_states,
                        class_labels=class_labels,
                        return_dict=False,
                    )[0]

                    if model_pred.shape[1] == 6:
                        model_pred, _ = torch.chunk(model_pred, 2, dim=1)

                    # Get the target for loss depending on the prediction type
                    if self.noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif self.noise_scheduler.config.prediction_type == "v_prediction":
                        target = self.noise_scheduler.get_velocity(model_input, noise, timesteps)
                    else:
                        raise ValueError(f"Unsupported prediction type: {self.noise_scheduler.config.prediction_type}")

                    if self.schema.with_prior_preservation:
                        # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
                        model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                        target, target_prior = torch.chunk(target, 2, dim=0)

                        # Compute prior loss
                        prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")

                    if not self.schema.snr_gamma:
                        # Compute the loss
                        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                    else:
                        loss = self._snr_gamma_weighting(model_pred, target, timesteps)

                    if self.schema.with_prior_preservation:
                        # Add the prior loss to the instance loss
                        loss = loss + self.schema.prior_loss_weight * prior_loss

                    self.accelerator.backward(loss)

                    if self.accelerator.sync_gradients:
                        params_to_clip = (
                            itertools.chain(self.unet.parameters(), self.text_encoder.parameters())
                            if self.schema.train_text_encoder
                            else self.unet.parameters()
                        )
                        self.accelerator.clip_grad_norm_(parameters=params_to_clip, max_norm=self.schema.max_grad_norm)

                    optimizer.step()
                    learning_rate_scheduler.step()
                    optimizer.zero_grad(set_to_none=self.schema.set_grads_to_none)

                # Checks if the accelerator has performed an optimization step behind the scenes
                if self.accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    if self.accelerator.is_main_process:
                        if global_step % self.schema.checkpointing_steps == 0:
                            if self.schema.checkpoints_total_limit:
                                self._handle_checkpoint_total_limit()

                            save_path = self.schema.output_dir / f"checkpoint-{global_step}"
                            self.accelerator.save_state(save_path)
                            self.logger.info(f"Saving checkpoint at {save_path}")

                        images = []

                        if self.schema.validation_prompt and global_step % self.schema.validation_steps == 0:
                            images = self._validate(global_step=global_step)

                logs = {"loss": loss.detach().item(), "lr": learning_rate_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                self.accelerator.log(logs, step=global_step)

                if global_step > self.schema.max_train_steps:
                    epoch_iterator.close()
                    break

        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            pipeline = self._save_trained_model()
            if self.schema.push_to_hub:
                self._save_model_card(images=images, pipeline=pipeline)
                self._upload_repo_to_hub()

        self.accelerator.end_training()
        self.logger.info("***** Training finished *****")
