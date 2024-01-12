from src.core.schemas.base import (
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


class DreamboothTrainerSchema(
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
    pass
