from pathlib import Path

from accelerate.utils.dataclasses import LoggerType
from diffusers.utils import is_tensorboard_available, is_wandb_available
from loguru import logger
from pydantic import BaseModel, Field, field_validator, model_validator

from src.core.constants import (
    LearningRateSchedulerEnum,
    OptimizerEnum,
    PrecisionTypeEnum,
    SchedulerEnum,
    VariantTypeEnum,
)

# TODO: think about moving these schemas into separate files


class ModelConfigurationSchema(BaseModel):
    """
    Schema for configuring the model. This includes the paths to pretrained models,
    the variant type of the model, and the tokenizer configuration.
    """

    pretrained_model_name_or_path: str = Field(
        description="Path to pretrained model or model identifier from huggingface.co/models."
    )
    pretrained_vae_name_or_path: str | None = Field(
        default=None,
        description=(
            "Path to pretrained VAE model with better numerical stability."
            " More details: https://github.com/huggingface/diffusers/pull/4038."
        ),
    )
    variant: VariantTypeEnum | None = Field(
        default=None,
        description="Variant of the model files of the pretrained model identifier from huggingface.co/models. for ex. 'fp16'",
    )
    tokenizer_name: str | None = Field(
        default=None,
        description="Pretrained tokenizer name or path if not the same as model_name",
    )


class InstanceSchema(BaseModel):
    """
    Schema for instance-level configurations. This includes specifying the instance prompt
    and the directory containing the instance-specific training images.
    """

    instance_prompt: str = Field(
        description="The prompt with identifier specifying the instance of the image to be generated."
    )
    instance_data_dir: Path = Field(description="Path to the directory containing the training instance images.")


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
    with_prior_preservation_loss: bool = Field(
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

    @model_validator(mode="after")
    def validate_with_prior_preservation_loss(self):
        if self.with_prior_preservation_loss:
            if not self.class_data_dir:
                error = "`class_data_dir` must be specified if `with_prior_preservation_loss` is True"
                logger.error(error)
                raise ValueError(error)
            if not self.class_prompt:
                error = "`class_prompt` must be specified if `with_prior_preservation_loss` is True"
                logger.error(error)
                raise ValueError(error)
        elif self.class_data_dir or self.class_prompt:
            logger.warning(
                "`class_data_dir` and class_prompt will be ignored as `with_prior_preservation_loss` is set to False."
            )
        return self


class DatasetSchema(BaseModel):
    """
    Schema for dataset configurations. Defines the dimensions of the input images,
    batch sizes for training and sampling, and dataloader worker settings.
    """

    width: int = Field(
        default=768,
        description=(
            "The width for input images, all the images in the train/validation dataset will be resized to this"
            " width"
        ),
    )
    height: int = Field(
        default=1024,
        description=(
            "The height for input images, all the images in the train/validation dataset will be resized to this"
            " height"
        ),
    )
    train_batch_size: int = Field(
        default=4,
        description="Batch size (per device) for the training dataloader.",
    )
    sample_batch_size: int = Field(
        default=4,
        description="Batch size (per device) for sampling images.",
    )
    dataloader_num_workers: int = Field(
        default=0,
        description=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )


class CheckpointManagementSchema(BaseModel):
    """
    Schema for managing training checkpoints. Includes configurations for checkpoint
    frequency, resuming from checkpoints, and the total limit of checkpoints stored.
    """

    checkpointing_steps: int = Field(
        default=500,
        description=(
            "Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`. "
            "In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference."
            "Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components."
            "See https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint for step by step"
            "instructions."
        ),
    )
    resume_from_checkpoint: Path | None = Field(
        default=None,
        description=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    checkpoints_total_limit: int | None = Field(
        default=None,
        description=(
            "Max number of checkpoints to store. Passed as `total_limit` to the `Accelerator` `ProjectConfiguration`."
            " See Accelerator::save_state https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.save_state"
            " for more details"
        ),
    )


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
    Schema for optimizer configurations. Defines parameters for the Adam optimizer
    including beta values, epsilon, weight decay, and optimizer type.
    """

    adam_beta1: float = Field(
        default=0.9,
        description="The beta1 parameter for the Adam optimizer.",
    )
    adam_beta2: float = Field(
        default=0.999,
        description="The beta2 parameter for the Adam optimizer.",
    )
    adam_epsilon: float = Field(
        default=1e-8,
        description="Epsilon value for the Adam optimizer",
    )
    weight_decay: float = Field(
        default=1e-2,
        description="Weight decay to use.",
    )
    optimizer: OptimizerEnum = Field(
        default=OptimizerEnum.ADAMW,
        description="Select which optimizer to use.",
    )


class HuggingFaceIntegrationSchema(BaseModel):
    """
    Schema for Hugging Face Hub integration. This includes settings for pushing the model
    to the hub, the hub token, and the model ID for the repository.
    """

    push_to_hub: bool = Field(
        description="Whether or not to push the model to the Hugging Face Hub after it's trained.",
    )
    hub_token: str | None = Field(
        default=None,
        description=(
            "The token to use to push to the Model Hub."
            " Ensure that the token is scoped to `write`."
            " See: https://huggingface.co/docs/hub/security-tokens for more information."
        ),
    )
    hub_model_id: str | None = Field(
        default=None,
        description="The name of the repository to keep in sync with the local `output_dir`.",
    )
    create_private_repo: bool = Field(
        default=True,
        description="Whether to create a private repository on the Hugging Face Hub.",
    )

    @model_validator(mode="after")
    def validate_hugging_face_integration(self):
        if self.push_to_hub and not self.hub_token:
            error = "`hub_token` must be specified if `push_to_hub` is True"
            logger.error(error)
            raise ValueError(error)
        return self


class LoggingSchema(BaseModel):
    """
    Schema for logging configurations. Includes settings for the logging directory
    and the platform to report logs and results to.
    """

    logging_dir: Path = Field(
        default="logs",
        description=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    report_to: LoggerType = Field(
        default=LoggerType.WANDB,
        description=('The integration to report the results and logs to. Use `"all"` to report to all integrations.'),
    )

    @field_validator("report_to", mode="after")
    @classmethod
    def validate_report_to(cls, value: LoggerType) -> LoggerType:
        if value == LoggerType.WANDB:
            if not is_wandb_available():
                error = "Make sure to install `wandb` (pip install wandb) if you want to use it for logging during training."
                logger.error(error)
                raise ValueError(error)

            import wandb

            if not wandb.api.api_key:
                error = "You must be logged in to Wandb(run: `wandb login`)"
                logger.error(error)
                raise ValueError(error)

        elif value == LoggerType.TENSORBOARD and not is_tensorboard_available():
            error = "Make sure to install `tensorboard` (pip install tensorboard) if you want to use it for logging during training."
            logger.error(error)
            raise ValueError(error)
        return value


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
        default=SchedulerEnum.EulerAncestralDiscreteScheduler,
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


class TrainingSetupSchema(BaseModel):
    """
    Schema for general training setup. Includes settings for output directory, seed,
    training epochs, steps, gradient norm, and tokenizer length.
    """

    output_dir: Path = Field(
        description="The output directory where the model predictions and checkpoints will be written.",
    )
    seed: int = Field(
        default=None,
        description="A seed for reproducible training.",
    )
    train_text_encoder: bool = Field(
        description="Whether to train the text encoder. If set, the text encoder should be float32 precision.",
    )
    text_encoder_use_attention_mask: bool = Field(
        default=False,
        description="Whether to use attention mask for text encoder.",
    )
    num_train_epochs: int = Field(
        default=100,
        description="Total number of training epochs to perform.",
    )
    max_train_steps: int | None = Field(
        default=None,
        description="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    max_grad_norm: float = Field(
        default=1.0,
        description="Max gradient norm.",
    )
    tokenizer_max_length: int | None = Field(
        default=None,
        description="The maximum length of the tokenizer. If not set, will default to the tokenizer's max length.",
    )


class AdvancedTrainingFeaturesSchema(BaseModel):
    """
    Schema for advanced training features. Includes configurations for offset noise,
    SNR gamma weighting, and other experimental training settings.
    """

    offset_noise: bool = Field(
        default=False,
        help=(
            "Fine-tuning against a modified noise"
            " See: https://www.crosslabs.org//blog/diffusion-with-offset-noise for more information."
        ),
    )
    snr_gamma: float | None = Field(
        default=None,
        description=(
            "SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0."
            " More details here: https://arxiv.org/abs/2303.09556."
        ),
    )
