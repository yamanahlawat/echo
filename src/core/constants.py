from enum import Enum


class BaseEnum(str, Enum):
    """
    An enum class that can get the value of an item with `str(Enum.key)`
    """

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))


class VariantTypeEnum(BaseEnum):
    """
    Enum for specifying the variant type of the model.
    """

    FP16 = "fp16"


class LearningRateSchedulerEnum(BaseEnum):
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
    COSINE_ANNEALING = "cosine_annealing"
    COSINE_ANNEALING_WARM_RESTARTS = "cosine_annealing_warm_restarts"


class SchedulerEnum(BaseEnum):
    """
    Enum for specifying the scheduler type.
    Includes different schedulers for controlling the learning rate and other training dynamics.
    """

    DPMSolverMultistepScheduler = "DPMSolverMultistepScheduler"
    DDPMScheduler = "DDPMScheduler"


class OptimizerEnum(BaseEnum):
    """
    Enum for specifying the optimizer type.
    """

    # TODO: Add support for more optimizers
    ADAMW = "adamw"
    ADAMW_8BIT = "adamw_8bit"
    LION = "lion"
    LION_8BIT = "lion_8bit"
    ADAFACTOR = "adafactor"


class ModelFileExtensions(BaseEnum):
    SAFETENSORS = ".safetensors"
    CKPT = ".ckpt"
    BIN = ".bin"
