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


class DreamboothTrainingSchema(
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
    ...
