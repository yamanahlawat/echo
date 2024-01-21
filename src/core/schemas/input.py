from pathlib import Path

from loguru import logger
from pydantic import BaseModel, Field, field_validator, model_validator

from src.core.constants import PrecisionTypeEnum


class InstanceSchema(BaseModel):
    """
    Schema for instance-level configurations. This includes specifying the instance prompt
    and the directory containing the instance-specific training images.
    """

    instance_prompt: str = Field(
        description="The prompt with identifier specifying the instance of the image to be generated."
    )
    instance_data_dir: Path = Field(description="Path to the directory containing the training instance images.")

    @field_validator("instance_data_dir", mode="after")
    def validate_instance_data_dir(cls, value: Path) -> Path:
        if not value.is_dir() and not value.exists():
            error = f"Instance data directory: '{value}' does not exist."
            logger.error(error)
            raise ValueError(error)
        return value


class ClassSchema(BaseModel):
    """
    Schema for class-level configurations. This includes class prompts, the directory
    for class-specific images, and settings related to prior preservation loss.
    """

    class_prompt: str | None = Field(
        default=None,
        description="The prompt to specify images in the same class as provided instance images.",
    )
    class_data_dir: Path | None = Field(
        default=None,
        description="Path to the directory containing the training class images.",
    )
    with_prior_preservation: bool = Field(
        default=False,
        description="Whether to use prior preservation loss.",
    )
    prior_loss_weight: float = Field(
        default=1.0,
        description="Weight of the prior preservation loss.",
    )
    num_class_images: int = Field(
        default=100,
        description=(
            "Minimal class images for prior preservation loss. If there are not enough images already present in"
            " class_data_dir, additional images will be sampled with class_prompt."
        ),
    )
    prior_generation_precision: PrecisionTypeEnum = Field(
        default=PrecisionTypeEnum.NO,
        description=(
            "Choose prior generation precision between fp32, fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10. and an Nvidia Ampere GPU."
        ),
    )
    class_labels_conditioning: str | None = Field(
        default=None,
        description="The optional `class_label` conditioning to pass to the unet, available values are `timesteps`.",
    )

    @model_validator(mode="after")
    def validate_with_prior_preservation(self):
        if self.with_prior_preservation:
            if not self.class_data_dir:
                error = "`class_data_dir` must be specified if `with_prior_preservation` is True"
                logger.error(error)
                raise ValueError(error)
            if not self.class_prompt:
                error = "`class_prompt` must be specified if `with_prior_preservation` is True"
                logger.error(error)
                raise ValueError(error)
        elif self.class_data_dir or self.class_prompt:
            logger.warning(
                "`class_data_dir` and class_prompt will be ignored as `with_prior_preservation` is set to False."
            )

        if self.class_data_dir and not self.class_data_dir.exists():
            logger.warning(f"`class_data_dir` does not exist: {self.class_data_dir}, creating it.")
            self.class_data_dir.mkdir(parents=True, exist_ok=True)
        return self
