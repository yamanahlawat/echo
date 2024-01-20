import logging

import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.utils.logging import set_verbosity_error, set_verbosity_info
from diffusers.utils.torch_utils import is_compiled_module
from huggingface_hub import create_repo
from loguru import logger as train_logger
from pydantic import BaseModel

from src.core.constants import ModelFileExtensions, OptimizerEnum

logger = get_logger(__name__)


class BaseTrainer:
    def __init__(self, schema: BaseModel) -> None:
        self.schema = schema
        # we must initialize the accelerate state before using the logging utility.
        self.accelerator = self._init_accelerator()
        self.weight_dtype = self._get_weight_dtype()
        self._init_logging()
        self.logger = logger
        self._init_seed()
        self.repo_id = self._create_model_repo()
        self.noise_scheduler = self._init_noise_scheduler()
        self.vae = self._init_vae()
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
            if vae_name.suffix in ModelFileExtensions.list():
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
        vae_dtype = torch.float32 if self.schema.no_half_vae else self.weight_dtype
        # TODO: check when to move to cuda
        vae.to(device=self.accelerator.device, dtype=vae_dtype)
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
            return torch.optim.AdamW(
                params=parameter_to_optimize,
                lr=self.schema.learning_rate,
                betas=(self.schema.beta1, self.schema.beta2),
                weight_decay=self.schema.weight_decay,
                eps=self.schema.epsilon,
            )
        elif self.schema.optimizer == OptimizerEnum.ADAMW_8BIT:
            from bitsandbytes.optim import Adam8bit

            return Adam8bit(
                params=parameter_to_optimize,
                lr=self.schema.learning_rate,
                betas=(self.schema.beta1, self.schema.beta2),
                weight_decay=self.schema.weight_decay,
                eps=self.schema.epsilon,
            )
        elif self.schema.optimizer == OptimizerEnum.LION:
            from bitsandbytes.optim import Lion

            return Lion(
                params=parameter_to_optimize,
                lr=self.schema.learning_rate,
                betas=(self.schema.beta1, self.schema.beta2),
                weight_decay=self.schema.weight_decay,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.schema.optimizer}")

    def _init_learning_rate_scheduler(self, optimizer):
        num_warmup_steps = self.schema.learning_rate_warmup_steps * self.accelerator.num_processes
        num_training_steps = self.schema.max_train_steps * self.accelerator.num_processes
        logger.info(
            f"Initializing learning rate scheduler: {self.schema.learning_rate_scheduler}, warmup steps: {num_warmup_steps}, training steps: {num_training_steps}"
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
        if self.schema.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif self.schema.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
        return weight_dtype

    def _unwrap_model(self, model):
        model = self.accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

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