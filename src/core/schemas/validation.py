from pydantic import BaseModel, Field

from src.core.constants import SchedulerEnum


class ValidationSchema(BaseModel):
    """
    Schema for validation configurations. Includes settings for validation prompts,
    number of validation images, validation steps, and related scheduler settings.
    """

    validation_prompt: str = Field(
        description="A prompt that is used during validation to verify that the model is learning.",
    )
    validation_negative_prompt: str = Field(
        description="A negative prompt that is used during validation to verify that the model is learning.",
    )
    num_validation_images: int = Field(
        default=5,
        description="Number of images that should be generated during validation with `validation_prompt`.",
    )
    validation_steps: int = Field(
        default=100,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `validation_prompt` multiple times: `num_validation_images`"
            " and logging the images."
        ),
    )
    validation_scheduler: SchedulerEnum = Field(
        default=SchedulerEnum.DPMSolverMultistepScheduler,
        description="Select which scheduler to use for validation. DDPMScheduler is recommended for DeepFloyd IF.",
    )
    validation_num_inference_steps: int = Field(
        default=50,
        description="The number of inference steps used for validation.",
    )
    validation_guidance_scale: float = Field(
        default=7.5,
        description="The guidance scale for classifier-free guidance for validation.",
    )
