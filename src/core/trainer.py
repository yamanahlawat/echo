import logging

import diffusers
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from huggingface_hub import create_repo
from loguru import logger as train_logger
from pydantic import BaseModel

from src.core.constants import ModelFileExtensions

logger = get_logger(__name__)


class BaseTrainer:
    def __init__(self, schema: BaseModel) -> None:
        self.schema = schema
        # we must initialize the accelerate state before using the logging utility.
        self.accelerator = self._init_accelerator()
        self._init_logging()
        self.logger = logger
        self._init_seed()
        self.repo_id = self._create_model_repo()
        self.noise_scheduler = self._init_noise_scheduler()
        self.vae = self._init_vae()
        self.unet = self._init_unet()

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
            diffusers.utils.logging.set_verbosity_info()
        else:
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()

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
                return AutoencoderKL.from_single_file(pretrained_model_link_or_path=vae_name)
            else:
                self.logger.info(f"Initializing VAE from pretrained VAE model: {vae_name}")
                return AutoencoderKL.from_pretrained(
                    pretrained_model_name_or_path=self.schema.pretrained_vae_name_or_path,
                    subfolder=sub_folder,
                    variant=self.schema.variant,
                )
        else:
            self.logger.info(f"Initializing VAE from pretrained model: {self.schema.pretrained_model_name_or_path}")
            return AutoencoderKL.from_pretrained(
                pretrained_model_name_or_path=self.schema.pretrained_model_name_or_path,
                subfolder=sub_folder,
                variant=self.schema.variant,
            )

    def _init_unet(self, sub_folder="unet"):
        self.logger.info(f"Initializing UNet from pretrained model: {self.schema.pretrained_model_name_or_path}")
        return UNet2DConditionModel.from_pretrained(
            pretrained_model_name_or_path=self.schema.pretrained_model_name_or_path,
            subfolder=sub_folder,
            variant=self.schema.variant,
        )
