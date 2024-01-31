import logging
from pathlib import Path

import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from accelerate.utils.dataclasses import PrecisionType
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.utils.logging import set_verbosity_error, set_verbosity_info
from huggingface_hub import create_repo, upload_folder
from loguru import logger as train_logger
from pydantic import BaseModel
from tqdm.auto import tqdm

from src.core.constants import LearningRateSchedulerEnum, ModelFileExtensions, OptimizerEnum
from src.core.dataset import PromptDataset
from src.core.factory import optimizers
from src.core.factory.lr_schedulers import get_cosine_annealing_scheduler, get_cosine_annealing_warm_restarts_scheduler

logger = get_logger(__name__)


class BaseTrainer:
    def __init__(self, schema: BaseModel) -> None:
        self.schema = schema
        self._init_seed()
        # we must initialize the accelerate state before using the logging utility.
        self.accelerator = self._init_accelerator()
        self.weight_dtype = self._get_weight_dtype()
        self.vae_dtype = torch.float32 if self.schema.no_half_vae else self.weight_dtype
        self._init_logging()
        self.logger = logger
        self._generate_class_images()
        self.repo_id = self._create_model_repo()
        self.noise_scheduler = self._init_noise_scheduler()
        self.vae = self._init_vae()
        self.vae_scaling_factor = self.vae.config.scaling_factor
        self.unet = self._init_unet()

        # Enable TF32 for faster training on Ampere GPUs,
        # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
        self._allow_tf32()

    def _init_accelerator(self):
        # using train_logger since logger from accelerate is not initialized
        train_logger.info("Initializing Accelerator...")
        logging_dir = self.schema.output_dir / self.schema.logging_dir
        accelerator_project_config = ProjectConfiguration(project_dir=self.schema.output_dir, logging_dir=logging_dir)
        return Accelerator(
            mixed_precision=self.schema.mixed_precision,
            gradient_accumulation_steps=self.schema.gradient_accumulation_steps,
            log_with=self.schema.report_to,
            project_config=accelerator_project_config,
        )

    def _init_logging(self):
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%d/%m/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logger.info(self.accelerator.state, main_process_only=False)
        if self.accelerator.is_local_main_process:
            transformers.utils.logging.set_verbosity_warning()
            set_verbosity_info()
        else:
            transformers.utils.logging.set_verbosity_error()
            set_verbosity_error()

    def _init_seed(self):
        if self.schema.seed:
            logger.info("Setting reproducible Training seed: {self.schema.seed}")
            set_seed(seed=self.schema.seed)

    def _generate_class_images(self):
        existing_class_images = len(list(self.schema.class_data_dir.iterdir()))
        if existing_class_images < self.schema.num_class_images:
            num_new_images = self.schema.num_class_images - existing_class_images
            logger.info(f"Generating {num_new_images} additional class images using prompt: {self.schema.class_prompt}")
            torch_dtype = torch.float16 if self.accelerator.device.type == "cuda" else torch.float32
            if self.schema.prior_generation_precision == "fp32":
                torch_dtype = torch.float32
            elif self.schema.prior_generation_precision == "fp16":
                torch_dtype = torch.float16
            elif self.schema.prior_generation_precision == "bf16":
                torch_dtype = torch.bfloat16
            pipeline = StableDiffusionPipeline.from_pretrained(
                pretrained_model_name_or_path=self.schema.pretrained_model_name_or_path,
                torch_dtype=torch_dtype,
                variant=self.schema.variant,
                safety_checker=None,
            )
            pipeline.set_progress_bar_config(disable=True)
            sample_dataset = PromptDataset(prompt=self.schema.class_prompt, num_samples=num_new_images)
            sample_dataloader = torch.utils.data.DataLoader(
                sample_dataset,
                batch_size=self.schema.sample_batch_size,
            )
            sample_dataloader = self.accelerator.prepare(sample_dataloader)
            pipeline.to(device=self.accelerator.device)

            for item in tqdm(
                sample_dataloader, desc="Generating class images", disable=not self.accelerator.is_local_main_process
            ):
                images = pipeline(prompt=item["prompt"]).images
                for i, image in enumerate(images):
                    image.save(self.schema.class_data_dir / f"image_{i}.png")

            del pipeline
            torch.cuda.empty_cache()
        else:
            logger.info(
                f"Found {existing_class_images} class images, which is more than the required {self.schema.num_class_images}."
            )

    def _create_model_repo(self):
        if self.schema.push_to_hub:
            repo_id = self.schema.hub_model_id or self.schema.output_dir.name
            logger.info(f"Creating model repo with id: {repo_id}")
            repo = create_repo(
                repo_id=repo_id,
                token=self.schema.hub_token,
                private=self.schema.create_private_repo,
                exist_ok=True,
            )
            logger.info(f"Model repo created with id: {repo.repo_id}, name: {repo.repo_name}")
            return repo.repo_id
        else:
            logger.warn("Model will not be pushed to the Huggingface.")

    def _init_noise_scheduler(self, sub_folder: str = "scheduler"):
        self.logger.info(
            f"Initializing DDPMScheduler noise scheduler from pretrained config {self.schema.pretrained_model_name_or_path}"
        )
        return DDPMScheduler.from_pretrained(
            pretrained_model_name_or_path=self.schema.pretrained_model_name_or_path,
            subfolder=sub_folder,
        )

    def _init_vae(self, sub_folder: str = "vae"):
        if vae_name := self.schema.pretrained_vae_name_or_path:
            if Path(vae_name).suffix in ModelFileExtensions.list():
                self.logger.info(f"Initializing VAE from file: {vae_name}")
                vae = AutoencoderKL.from_single_file(pretrained_model_link_or_path=vae_name)
            else:
                self.logger.info(f"Initializing VAE from pretrained VAE model: {vae_name}")
                vae = AutoencoderKL.from_pretrained(
                    pretrained_model_name_or_path=self.schema.pretrained_vae_name_or_path,
                    subfolder=sub_folder,
                    variant=self.schema.variant,
                )
        else:
            self.logger.info(f"Initializing VAE from pretrained model: {self.schema.pretrained_model_name_or_path}")
            vae = AutoencoderKL.from_pretrained(
                pretrained_model_name_or_path=self.schema.pretrained_model_name_or_path,
                subfolder=sub_folder,
                variant=self.schema.variant,
            )
        if vae:
            vae.requires_grad_(False)
        vae.to(device=self.accelerator.device, dtype=self.vae_dtype)
        return vae

    def _init_unet(self, sub_folder="unet"):
        self.logger.info(f"Initializing UNet from pretrained model: {self.schema.pretrained_model_name_or_path}")
        unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_name_or_path=self.schema.pretrained_model_name_or_path,
            subfolder=sub_folder,
            variant=self.schema.variant,
        )
        if unet.dtype != torch.float32:
            raise ValueError(
                f"Unet loaded as datatype {unet.dtype}. Please make sure to always have all model weights in full float32 precision."
            )
        return unet

    def _allow_tf32(self):
        if self.schema.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True

    def _init_optimizer(self, parameter_to_optimize):
        if self.schema.optimizer == OptimizerEnum.ADAMW:
            return optimizers.get_adamw_optimizer(
                params=parameter_to_optimize,
                learning_rate=self.schema.learning_rate,
                betas=(self.schema.beta1, self.schema.beta2),
                weight_decay=self.schema.weight_decay,
                eps=self.schema.epsilon,
            )
        elif self.schema.optimizer == OptimizerEnum.ADAMW_8BIT:
            return optimizers.get_adam8bit_optimizer(
                params=parameter_to_optimize,
                learning_rate=self.schema.learning_rate,
                betas=(self.schema.beta1, self.schema.beta2),
                weight_decay=self.schema.weight_decay,
                eps=self.schema.epsilon,
            )
        elif self.schema.optimizer == OptimizerEnum.LION:
            return optimizers.get_lion_optimizer(
                params=parameter_to_optimize,
                learning_rate=self.schema.learning_rate,
                betas=(self.schema.beta1, self.schema.beta2),
                weight_decay=self.schema.weight_decay,
            )
        elif self.schema.optimizer == OptimizerEnum.LION_8BIT:
            return optimizers.get_lion8bit_optimizer(
                params=parameter_to_optimize,
                learning_rate=self.schema.learning_rate,
                betas=(self.schema.beta1, self.schema.beta2),
                weight_decay=self.schema.weight_decay,
            )
        elif self.schema.optimizer == OptimizerEnum.ADAFACTOR:
            return optimizers.get_adafactor_optimizer(
                params=parameter_to_optimize,
                learning_rate=self.schema.learning_rate,
                weight_decay=self.schema.weight_decay,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.schema.optimizer}")

    def _init_learning_rate_scheduler(self, optimizer):
        num_warmup_steps = self.schema.learning_rate_warmup_steps * self.accelerator.num_processes
        num_training_steps = self.schema.max_train_steps * self.accelerator.num_processes

        if self.schema.learning_rate_scheduler == LearningRateSchedulerEnum.COSINE_ANNEALING_WARM_RESTARTS:
            return get_cosine_annealing_warm_restarts_scheduler(optimizer=optimizer, T_0=num_warmup_steps)

        elif self.schema.learning_rate_scheduler == LearningRateSchedulerEnum.COSINE_ANNEALING:
            return get_cosine_annealing_scheduler(optimizer=optimizer, T_max=num_training_steps)

        else:
            logger.info(
                f"Initializing learning rate scheduler: {self.schema.learning_rate_scheduler}, warmup steps: {num_warmup_steps},"
                f" training steps: {num_training_steps}"
            )
            return get_scheduler(
                name=self.schema.learning_rate_scheduler,
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
            )

    def _get_weight_dtype(self):
        # For mixed precision training we cast all non-trainable weights
        # (vae, non-lora text_encoder and non-lora unet) to half-precision
        # as these weights are only used for inference, keeping weights in full precision is not required.
        weight_dtype = torch.float32
        if self.accelerator.mixed_precision == PrecisionType.FP16.value:
            weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == PrecisionType.BF16.value:
            weight_dtype = torch.bfloat16
        return weight_dtype

    def _get_all_checkpoints_paths(self):
        return sorted(self.schema.output_dir.glob("checkpoint-*"))

    def _resume_from_checkpoint(self, num_update_steps_per_epoch):
        if self.schema.resume_from_checkpoint == "latest":
            checkpoint_path = self._get_all_checkpoints_paths()[-1]
        else:
            checkpoint_path = self.schema.resume_from_checkpoint
        # TODO: check if checkpoint doesn't exist should we start the training from start or raise error
        if not checkpoint_path.exists():
            raise ValueError(f"Checkpoint path `{self.schema.resume_from_checkpoint}` does not exist.")
        logger.info(f"Resuming training from checkpoint: {checkpoint_path}")
        self.accelerator.load_state(checkpoint_path)
        global_step = int(checkpoint_path.stem.split("-")[-1])
        first_epoch = global_step // num_update_steps_per_epoch
        # since initial global step is same as global step, we return it twice
        return global_step, global_step, first_epoch

    def _handle_checkpoint_total_limit(self):
        if self.schema.checkpoints_total_limit:
            all_checkpoints_paths = self._get_all_checkpoints_paths()
            if len(all_checkpoints_paths) > self.schema.checkpoints_total_limit:
                no_of_checkpoints_to_remove = len(all_checkpoints_paths) - self.schema.checkpoints_total_limit + 1
                checkpoints_to_remove = all_checkpoints_paths[:no_of_checkpoints_to_remove]
                logger.info(
                    f"Total Checkpoints: {len(all_checkpoints_paths)}, Removing checkpoints: {checkpoints_to_remove}"
                )
                for checkpoint in checkpoints_to_remove:
                    checkpoint.unlink()

    def _get_scheduler_args(self, pipeline):
        # We train on the simplified learning objective. If we were previously predicting a variance,
        # we need the scheduler to ignore it
        scheduler_args = {}

        if "variance_type" in pipeline.scheduler.config:
            variance_type = pipeline.scheduler.config.variance_type

            if variance_type in ["learned", "learned_range"]:
                variance_type = "fixed_small"

            scheduler_args["variance_type"] = variance_type
        return scheduler_args

    def _generate_gallery_widget(self, images):
        widget_str = "widget:\n"
        for i, image in enumerate(images):
            image_path = self.schema.output_dir / f"image_{i}.png"
            image.save(image_path)
            widget_str += f"- text: '{self.schema.validation_prompt}'\n"
            widget_str += f"  output:\n    url: './image_{i}.png'\n"
        return widget_str

    def _save_model_card(self, images, pipeline):
        gallery_widget = self._generate_gallery_widget(images)

        tags = [
            "stable-diffusion" if isinstance(pipeline, StableDiffusionPipeline) else "if",
            "stable-diffusion-diffusers" if isinstance(pipeline, StableDiffusionPipeline) else "if-diffusers",
            "text-to-image",
            "diffusers",
            "dreambooth",
        ]
        tags_list = "\n".join(f"- {tag}" for tag in tags)

        repo_config = (
            "---\n"
            "license: creativeml-openrail-m\n"
            f"base_model: {self.schema.pretrained_model_name_or_path}\n"
            f"instance_prompt: {self.schema.instance_prompt}\n"
            "library_name: echo\n"
            "tags:\n"
            f"{tags_list}\n"
            "inference: true\n"
            f"{gallery_widget}\n"
            "---\n\n"
        )

        model_card_content = (
            f"# DreamBooth - {self.repo_id}\n\n"
            "<Gallery />\n\n"
            f"This is a dreambooth model derived from {self.schema.pretrained_model_name_or_path}.\n"
            f"The weights were trained on {self.schema.instance_prompt} using [DreamBooth](https://dreambooth.github.io/).\n\n"
            f"DreamBooth for the text encoder was enabled: {self.schema.train_text_encoder}\n"
        )

        with open(self.schema.output_dir / "README.md", "w") as readme_file:
            readme_file.write(repo_config + model_card_content)

    def _upload_repo_to_hub(self):
        upload_folder(
            repo_id=self.repo_id,
            folder_path=self.schema.output_dir,
            commit_message=f"Trained DreamBooth model {self.schema.instance_prompt}",
            ignore_patterns=["step_*", "epoch_*"],
            token=self.schema.hub_token,
        )

    def _cache_latents(self, train_dataloader):
        cached_latents = []
        for batch in tqdm(train_dataloader, desc="Caching latents"):
            with torch.no_grad():
                latents = batch["pixel_values"].to(dtype=self.vae_dtype)
                latents = self.vae.encode(latents).latent_dist
                cached_latents.append(latents)

        # Move vae to CPU
        logger.info("Moving VAE to CPU")
        self.vae = self.vae.to("cpu")
        torch.cuda.empty_cache()
        return cached_latents
