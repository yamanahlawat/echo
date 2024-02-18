from pathlib import Path

from loguru import logger
from pydantic import BaseModel, Field, field_validator, model_validator
from slugify import slugify


class TrainingSetupSchema(BaseModel):
    """
    Schema for general training setup. Includes settings for output directory, seed,
    training epochs, steps, gradient norm, and tokenizer length.
    """

    output_dir: Path = Field(
        description="The output directory where the model predictions and checkpoints will be written.",
    )
    save_safetensors: bool = Field(
        default=False,
        description="Whether to save a file .safetensors in the output directory.",
    )
    seed: int = Field(
        default=None,
        description="A seed for reproducible training.",
    )
    train_text_encoder: bool = Field(
        description="Whether to train the text encoder. If set, the text encoder should be float32 precision.",
    )
    pre_compute_text_embeddings: bool = Field(
        description=(
            "Whether or not to pre-compute text embeddings. If text embeddings are pre-computed,"
            " the text encoder will not be kept in memory during training and will leave more GPU memory available for training the rest of the model."
            " This is not compatible with `--train_text_encoder`."
        ),
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

    @field_validator("output_dir")
    @classmethod
    def validate_output_dir(cls, value: Path) -> Path:
        value = value.parent / slugify(value.name)
        if not value.exists():
            logger.info(f"Output directory: '{value}' does not exist. Creating it.")
            value.mkdir(parents=True, exist_ok=True)
        return value

    @model_validator(mode="after")
    def validate_training_setup_params(self):
        # validate train_text_encoder and pre_compute_text_embeddings
        if self.train_text_encoder and self.pre_compute_text_embeddings:
            error = "`train_text_encoder` and `pre_compute_text_embeddings` are not compatible."
            logger.error(error)
            raise ValueError(error)
        return self


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
