from enum import Enum


class VariantTypeEnum(str, Enum):
    """
    Enum for specifying the variant type of the model.
    """

    FP16 = "fp16"


class LearningRateSchedulerEnum(str, Enum):
    """
    Enum for specifying the learning rate scheduler type.
    - CONSTANT: Keeps the learning rate constant.
    - LINEAR: Linearly decreases the learning rate.
    - COSINE: Cosine annealing scheduler.
    - COSINE_WITH_RESTARTS: Cosine annealing scheduler with restarts.
    - POLYNOMIAL: Polynomial decay of the learning rate.
    - CONSTANT_WITH_WARMUP: Constant learning rate with a warmup period.
    """

    CONSTANT = "constant"
    LINEAR = "linear"
    COSINE = "cosine"
    COSINE_WITH_RESTARTS = "cosine_with_restarts"
    POLYNOMIAL = "polynomial"
    CONSTANT_WITH_WARMUP = "constant_with_warmup"


class PrecisionTypeEnum(str, Enum):
    """
    Enum for specifying the precision type for training.
    - NO: No specific precision.
    - FP16: 16-bit floating-point precision.
    - BF16: Brain floating-point format with 16-bit precision.
    - FP32: 32-bit floating-point precision.
    """

    NO = "no"
    FP16 = "fp16"
    BF16 = "bf16"
    FP32 = "fp32"


class SchedulerEnum(str, Enum):
    """
    Enum for specifying the scheduler type.
    Includes different schedulers for controlling the learning rate and other training dynamics.
    """

    DPMSolverMultistepScheduler = "DPMSolverMultistepScheduler"
    DDPMScheduler = "DDPMScheduler"
    EulerAncestralDiscreteScheduler = "EulerAncestralDiscreteScheduler"


class OptimizerEnum(str, Enum):
    """
    Enum for specifying the optimizer type.
    - ADAMW: Adam optimizer with weight decay.
    - LION: LION optimizer.
    """

    # TODO: Add support for more optimizers
    ADAMW = "adamw"
    LION = "lion"
