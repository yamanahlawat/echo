from loguru import logger
from pydantic import BaseModel, Field, field_validator

from src.core.constants import LearningRateSchedulerEnum, OptimizerEnum


class LearningRateSchema(BaseModel):
    """
    Schema for learning rate configurations. This includes the initial learning rate,
    scaling options, and various scheduler settings.
    """

    learning_rate: float = Field(
        default=5e-6,
        description="Initial learning rate (after the potential warmup period) to use.",
    )
    scale_learning_rate: bool = Field(
        default=False,
        description="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    learning_rate_scheduler: LearningRateSchedulerEnum = Field(
        default=LearningRateSchedulerEnum.CONSTANT,
        description="The scheduler type to use.",
    )
    learning_rate_warmup_steps: int = Field(
        default=500,
        description="Number of steps for the warmup in the learning rate scheduler.",
    )
    learning_rate_num_cycles: int = Field(
        default=1,
        description="Number of hard resets of the learning rate in cosine_with_restarts scheduler.",
    )
    learning_rate_power: float = Field(default=1.0, description="Power factor of the polynomial scheduler.")


class OptimizerSchema(BaseModel):
    """
    Schema for optimizer configurations. Defines parameters for the optimizer
    including beta values, epsilon, weight decay, and optimizer type.
    """

    beta1: float = Field(
        default=0.9,
        description="The beta1 parameter for the optimizer.",
    )
    beta2: float = Field(
        default=0.999,
        description="The beta2 parameter for the optimizer.",
    )
    epsilon: float = Field(
        default=1e-8,
        description="Epsilon value for the optimizer",
    )
    weight_decay: float = Field(
        default=1e-2,
        description="Weight decay to use.",
    )
    optimizer: OptimizerEnum = Field(
        default=OptimizerEnum.ADAMW,
        description="Select which optimizer to use.",
        validate_default=True,
    )

    @field_validator("optimizer", mode="after")
    @classmethod
    def validate_optimizer(cls, value: OptimizerEnum) -> OptimizerEnum:
        if value == OptimizerEnum.ADAMW_8BIT:
            try:
                import bitsandbytes  # noqa: F401
            except ImportError as e:
                error = "Make sure to install `bitsandbytes` (pip install bitsandbytes) if you want to use it for 8-bit optimizer."
                logger.error(error)
                raise ValueError(error) from e
        if value == OptimizerEnum.LION:
            try:
                # TODO: check whether to use Lion from pytorch_optimizer or from bitsandbytes
                from pytorch_optimizer import Lion  # noqa: F401
            except ImportError as e:
                error = "Make sure to install `pytorch_optimizer` (pip install pytorch-optimizer) if you want to use it for Lion optimizer."
                logger.error(error)
                raise ValueError(error) from e
        return value
