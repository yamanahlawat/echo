import os
from pathlib import Path

from accelerate.utils.dataclasses import LoggerType
from diffusers.utils.import_utils import is_tensorboard_available, is_wandb_available
from loguru import logger
from pydantic import BaseModel, Field, SecretStr, model_validator


class LoggingSchema(BaseModel):
    """
    Schema for logging configurations. Includes settings for the logging directory
    and the platform to report logs and results to.
    """

    logging_dir: Path = Field(
        default="logs",
        description=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    report_to: LoggerType = Field(
        default=LoggerType.WANDB,
        description='The integration to report the results and logs to. Use `"all"` to report to all integrations.',
        validate_default=True,
    )
    wandb_api_key: SecretStr | None = Field(
        default=None,
        description="The API key to use for logging to W&B. If not specified, will default to the environment variable `WANDB_API_KEY`.",
    )

    @model_validator(mode="after")
    def validate_logging_schema(self):
        if self.report_to == LoggerType.WANDB:
            if not is_wandb_available():
                error = "Make sure to install `wandb` (pip install wandb) if you want to use it for logging during training."
                logger.error(error)
                raise ValueError(error)

            import wandb

            if not (wandb_api_key := self.wandb_api_key.get_secret_value() or os.getenv("WANDB_API_KEY")):
                raise ValueError(
                    "You must provide a wandb_api_key or have WANDB_API_KEY set in your environment variables."
                    "or You must be logged in to Wandb(run: `wandb login`)"
                )
            try:
                wandb.login(key=wandb_api_key, verify=True)
            except Exception as error:
                raise ValueError(f"Failed to login to wandb with the provided API key: {error}") from error

        elif self.report_to == LoggerType.TENSORBOARD and not is_tensorboard_available():
            error = "Make sure to install `tensorboard` (pip install tensorboard) if you want to use it for logging during training."
            logger.error(error)
            raise ValueError(error)
        return self
