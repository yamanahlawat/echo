from diffusers.utils.import_utils import is_xformers_available
from loguru import logger
from pydantic import BaseModel, Field, field_validator

from src.core.constants import PrecisionTypeEnum


class ResourceOptimizationSchema(BaseModel):
    """
    Schema for resource optimization configurations. Includes mixed precision settings,
    memory-efficient attention options, gradient accumulation steps, and others.
    """

    mixed_precision: PrecisionTypeEnum = Field(
        default=PrecisionTypeEnum.NO,
        description=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    enable_xformers_memory_efficient_attention: bool = Field(
        default=False,
        description="Whether or not to use xformers.",
    )
    set_grads_to_none: bool = Field(
        default=False,
        description=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )
    gradient_accumulation_steps: int = Field(
        default=1,
        description="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    gradient_checkpointing: bool = Field(
        default=False,
        description="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    allow_tf32: bool = Field(
        default=False,
        description=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )

    @field_validator("enable_xformers_memory_efficient_attention", mode="after")
    @classmethod
    def validate_enable_xformers_memory_efficient_attention(cls, value: bool) -> bool:
        if value and not is_xformers_available():
            error = "Make sure to install `xformers` (pip install xformers) if you want to use it for memory efficient attention during training."
            logger.error(error)
            raise ValueError(error)
        return value
