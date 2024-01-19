from pydantic import BaseModel, Field

from src.core.constants import VariantTypeEnum


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
    no_half_vae: bool = Field(
        default=False,
        description="Do not use half precision for the VAE.",
    )
    variant: VariantTypeEnum | None = Field(
        default=None,
        description="Variant of the model files of the pretrained model identifier from huggingface.co/models. for ex. 'fp16'",
    )
    tokenizer_name: str | None = Field(
        default=None,
        description="Pretrained tokenizer name or path if not the same as model_name",
    )
