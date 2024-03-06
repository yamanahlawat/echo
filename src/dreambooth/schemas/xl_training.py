from loguru import logger
from pydantic import field_validator

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
    @field_validator("tokenizer_name", mode="after")
    @classmethod
    def validate_tokenizer_name(cls, value: str) -> str:
        if value:
            error = "We currently do not support loading tokenizer for sdxl trainer"
            logger.error(error)
            raise ValueError(error)
        return value
