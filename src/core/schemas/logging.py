from pathlib import Path

from accelerate.utils.dataclasses import LoggerType
from diffusers.utils.import_utils import is_tensorboard_available, is_wandb_available
from loguru import logger
from pydantic import BaseModel, Field, field_validator


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

    @field_validator("report_to", mode="after")
    @classmethod
    def validate_report_to(cls, value: LoggerType) -> LoggerType:
        if value == LoggerType.WANDB:
            if not is_wandb_available():
                error = "Make sure to install `wandb` (pip install wandb) if you want to use it for logging during training."
                logger.error(error)
                raise ValueError(error)

            import wandb

            if not wandb.api.api_key:
                error = "You must be logged in to Wandb(run: `wandb login`)"
                logger.error(error)
                raise ValueError(error)

        elif value == LoggerType.TENSORBOARD and not is_tensorboard_available():
            error = "Make sure to install `tensorboard` (pip install tensorboard) if you want to use it for logging during training."
            logger.error(error)
            raise ValueError(error)
        return value
