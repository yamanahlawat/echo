from pathlib import Path

import torch
from diffusers import (
    AutoencoderKL,
    EulerDiscreteScheduler,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

from src.core.constants import ModelFileExtensions
from src.core.dataset import PromptDataset
from src.core.trainer import BaseTrainer
from src.dreambooth.schemas.xl_training import DreamboothXLTrainingSchema


class DreamboothXLTrainer(BaseTrainer):
    def __init__(self, schema: DreamboothXLTrainingSchema) -> None:
        super().__init__(schema=schema)
        self.logger.info("Initializing Dreambooth Stable Diffusion XL Trainer...")

        # Initialize the models
        self.pipeline = None
        if Path(self.schema.pretrained_model_name_or_path).suffix == ModelFileExtensions.SAFETENSORS.value:
            self.pipeline = self._init_pipeline()

        self.noise_scheduler = self._init_noise_scheduler()
        self.unet = self._init_unet()
        self.vae = self._init_vae()

        # Scale the image latents according to the vae scale factor
        self.vae_scaling_factor = self.vae.config.scaling_factor

        # tokenizer
        self.tokenizer_one, self.tokenizer_two = self._init_tokenizers()
        # list of all tokenizers
        self.tokenizers = [self.tokenizer_one, self.tokenizer_two]

        # text encoder
        if self.pipeline:
            self.text_encoder_model_class_one = self.pipeline.text_encoder.__class__
            self.text_encoder_model_class_two = self.pipeline.text_encoder_2.__class__
        else:
            self.text_encoder_model_class_one = self._get_text_encoder_model_class()
            self.text_encoder_model_class_one = self._get_text_encoder_model_class(sub_folder="text_encoder_2")
        self.text_encoder_one, self.text_encoder_two = self._init_text_encoders()
        # list of all text encoders
        self.text_encoders = [self.text_encoder_one, self.text_encoder_two]

    def _init_pipeline(self):
        self.logger.info(f"Loading safetensors pipeline from file: {self.schema.pretrained_model_name_or_path}")
        return StableDiffusionXLPipeline.from_single_file(
            pretrained_model_link_or_path=self.schema.pretrained_model_name_or_path,
            safety_checker=self.schema.safety_checker,
            variant=self.schema.variant,
        )

    def _init_noise_scheduler(self, sub_folder: str = "scheduler"):
        if self.pipeline:
            self.logger.info(f"Initializing noise scheduler from pipeline: {self.schema.pretrained_model_name_or_path}")
            return self.pipeline.scheduler
        self.logger.info(
            f"Initializing noise scheduler from pretrained model: {self.schema.pretrained_model_name_or_path}"
        )
        return EulerDiscreteScheduler.from_pretrained(
            pretrained_model_name_or_path=self.schema.pretrained_model_name_or_path,
            subfolder=sub_folder,
        )

    def _init_unet(self, sub_folder="unet"):
        if self.pipeline:
            self.logger.info(f"Initializing UNet from pipeline: {self.schema.pretrained_model_name_or_path}")
            unet = self.pipeline.unet
        else:
            self.logger.info(f"Initializing UNet from pretrained model: {self.schema.pretrained_model_name_or_path}")
            unet = UNet2DConditionModel.from_pretrained(
                pretrained_model_name_or_path=self.schema.pretrained_model_name_or_path,
                subfolder=sub_folder,
                variant=self.schema.variant,
            )

        # TODO: check if we upcast the model weights or raise an error
        # Configure UNet
        if unet.dtype != torch.float32:
            raise ValueError(
                f"Unet loaded as datatype {unet.dtype}. Please make sure to always have all model weights in full float32 precision."
            )
        if self.schema.enable_xformers_memory_efficient_attention:
            unet.enable_xformers_memory_efficient_attention()
        return unet

    def _init_vae(self, sub_folder="vae"):
        if vae_name := self.schema.pretrained_vae_name_or_path:
            if Path(vae_name).suffix in ModelFileExtensions.list():
                self.logger.info(f"Initializing VAE from file: {vae_name}")
                vae = AutoencoderKL.from_single_file(pretrained_model_link_or_path=vae_name)
            elif self.pipeline:
                self.logger.info(f"Initializing VAE from pipeline: {vae_name}")
                vae = self.pipeline.vae
            else:
                self.logger.info(f"Initializing VAE from pretrained VAE model: {vae_name}")
                vae = AutoencoderKL.from_pretrained(
                    pretrained_model_name_or_path=self.schema.pretrained_vae_name_or_path,
                    subfolder=sub_folder,
                    variant=self.schema.variant,
                )
        elif self.pipeline:
            self.logger.info(f"Initializing VAE from pipeline: {self.schema.pretrained_model_name_or_path}")
            vae = self.pipeline.vae
        else:
            self.logger.info(f"Initializing VAE from pretrained model: {self.schema.pretrained_model_name_or_path}")
            vae = AutoencoderKL.from_pretrained(
                pretrained_model_name_or_path=self.schema.pretrained_model_name_or_path,
                subfolder=sub_folder,
                variant=self.schema.variant,
            )
        # Configure VAE
        vae.requires_grad_(False)
        vae.to(device=self.accelerator.device, dtype=self.vae_dtype)
        return vae

    def _init_tokenizers(self, tokenizer_one_sub_folder="tokenizer", tokenizer_two_sub_folder="tokenizer_2"):
        if self.pipeline:
            self.logger.info(
                f"Initializing tokenizer and tokenizer_2 from pipeline: {self.schema.pretrained_model_name_or_path}"
            )
            return self.pipeline.tokenizer, self.pipeline.tokenizer_2

        self.logger.info(
            f"Initializing tokenizer and tokenizer_2 from pretrained model: {self.schema.pretrained_model_name_or_path}"
        )
        tokenizer_one = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.schema.pretrained_model_name_or_path,
            subfolder=tokenizer_one_sub_folder,
            use_fast=False,
        )
        tokenizer_two = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.schema.pretrained_model_name_or_path,
            subfolder=tokenizer_two_sub_folder,
            use_fast=False,
        )
        return tokenizer_one, tokenizer_two

    def _get_text_encoder_model_class(self, sub_folder: str = "text_encoder"):
        self.logger.info(
            f"Getting Text Encoder model class from pretrained model: {self.schema.pretrained_model_name_or_path}"
        )
        text_encoder_config = PretrainedConfig.from_pretrained(
            pretrained_model_name_or_path=self.schema.pretrained_model_name_or_path,
            subfolder=sub_folder,
        )
        model_class_name = text_encoder_config.architectures[0]
        if model_class_name == "CLIPTextModel":
            from transformers import CLIPTextModel

            return CLIPTextModel
        elif model_class_name == "CLIPTextModelWithProjection":
            from transformers import CLIPTextModelWithProjection

            return CLIPTextModelWithProjection
        else:
            ValueError(f"Unsupported Text Encoder model class: {model_class_name}")

    def _init_text_encoders(
        self, text_encoder_one_subfolder: str = "text_encoder", text_encoder_two_subfolder: str = "text_encoder_2"
    ):
        if self.pipeline:
            self.logger.info(
                f"Initializing text_encoder and text_encoder_2 from pipeline: {self.schema.pretrained_model_name_or_path}"
            )
            text_encoder_one, text_encoder_two = self.pipeline.text_encoder, self.pipeline.text_encoder_2
        else:
            self.logger.info(
                f"Initializing text_encoder and text_encoder_2 from pretrained model: {self.schema.pretrained_model_name_or_path}"
            )
            text_encoder_one = self.text_encoder_model_class_one.from_pretrained(
                pretrained_model_name_or_path=self.schema.pretrained_model_name_or_path,
                subfolder=text_encoder_one_subfolder,
            )
            text_encoder_two = self.text_encoder_model_class_two.from_pretrained(
                pretrained_model_name_or_path=self.schema.pretrained_model_name_or_path,
                subfolder=text_encoder_two_subfolder,
            )
        if self.schema.train_text_encoder and (
            text_encoder_one.dtype != torch.float32 or text_encoder_two.dtype != torch.float32
        ):
            raise ValueError(
                f"Text encoder one loaded as datatype {text_encoder_one.dtype}. Text encoder two loaded as datatype {text_encoder_two.dtype}."
                "Please make sure to always have all model weights in full float32 precision."
            )

        if not self.schema.train_text_encoder:
            text_encoder_one.requires_grad_(False)
            text_encoder_two.requires_grad_(False)
        return text_encoder_one, text_encoder_two

    def _generate_class_images(self):
        existing_class_images = len(list(self.schema.class_data_dir.iterdir()))
        if existing_class_images < self.schema.num_class_images:
            torch_dtype = torch.float16 if self.accelerator.device.type == "cuda" else torch.float32
            if self.schema.prior_generation_precision == "fp32":
                torch_dtype = torch.float32
            elif self.schema.prior_generation_precision == "fp16":
                torch_dtype = torch.float16
            elif self.schema.prior_generation_precision == "bf16":
                torch_dtype = torch.bfloat16
            if self.pipeline:
                pipeline = self.pipeline
            else:
                pipeline = StableDiffusionXLPipeline.from_pretrained(
                    pretrained_model_name_or_path=self.schema.pretrained_model_name_or_path,
                    torch_dtype=torch_dtype,
                    variant=self.schema.variant,
                    safety_checker=self.schema.safety_checkers,
                )
            pipeline.set_progress_bar_config(disable=True)
            num_new_images = self.schema.num_class_images - existing_class_images
            self.logger.info(
                f"Generating {num_new_images} additional class images using prompt: {self.schema.class_prompt}"
            )
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
                images = pipeline(
                    prompt=item["prompt"],
                    height=self.schema.height,
                    width=self.schema.width,
                ).images
                for i, image in enumerate(images):
                    image.save(self.schema.class_data_dir / f"image_{i}.png")

            del pipeline
            torch.cuda.empty_cache()
        else:
            self.logger.info(
                f"Found {existing_class_images} class images, which is more than the required {self.schema.num_class_images}."
            )

    def _tokenize_prompt(self, tokenizer: AutoTokenizer, prompt: str, add_special_tokens: bool = False):
        return tokenizer(
            text=prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            add_special_tokens=add_special_tokens,
            return_tensors="pt",
        )

    def _encode_prompt(
        self, tokenizers: list | None = None, prompt: str | None = None, text_input_ids_list: list | None = None
    ):
        prompt_embed_list = []

        for index, text_encoder in enumerate(self.text_encoders):
            if tokenizers:
                text_inputs = self._tokenize_prompt(tokenizer=tokenizers[index], prompt=prompt)
                text_input_ids = text_inputs.input_ids
            else:
                assert text_input_ids_list is not None
                text_input_ids = text_input_ids_list[index]

            prompt_embeds = text_encoder(
                input_ids=text_input_ids.to(device=self.accelerator.device),
                output_hidden_states=True,
                return_dict=False,
            )
            pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds[-1][-2]
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.view(bs_embed * seq_len, -1)
            prompt_embed_list.append(prompt_embeds)

        prompt_embeds = torch.cat(prompt_embed_list, dim=-1)
        pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
        return prompt_embeds, pooled_prompt_embeds

    def _compute_text_embeddings(self, prompt: str):
        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds = self._encode_prompt(tokenizers=self.tokenizers, prompt=prompt)
            prompt_embeds = prompt_embeds.to(device=self.accelerator.device)
            pooled_prompt_embeds = pooled_prompt_embeds.to(device=self.accelerator.device)
        return prompt_embeds, pooled_prompt_embeds
