from loguru import logger
from pydantic import Field, field_validator

from src.core.schemas import (
    AdvancedTrainingFeaturesSchema,
    CheckpointManagementSchema,
    ClassSchema,
    DatasetSchema,
    HuggingFaceIntegrationSchema,
    InstanceSchema,
    LearningRateSchema,
    LoggingSchema,
    ModelConfigurationSchema,
    OptimizerSchema,
    ResourceOptimizationSchema,
    TrainingSetupSchema,
    ValidationSchema,
)


class DreamboothXLTrainingSchema(
    AdvancedTrainingFeaturesSchema,
    TrainingSetupSchema,
    ResourceOptimizationSchema,
    ValidationSchema,
    LoggingSchema,
    HuggingFaceIntegrationSchema,
    OptimizerSchema,
    LearningRateSchema,
    CheckpointManagementSchema,
    DatasetSchema,
    ClassSchema,
    InstanceSchema,
    ModelConfigurationSchema,
):
    edm_style_training: bool = Field(
        default=False,
        description="Whether to conduct training using the EDM formulation as introduced in https://arxiv.org/abs/2206.00364.",
    )

    @field_validator("tokenizer_name", mode="after")
    @classmethod
    def validate_tokenizer_name(cls, value: str) -> str:
        if value:
            error = "We currently do not support loading tokenizer for sdxl trainer"
            logger.error(error)
            raise ValueError(error)
        return value
