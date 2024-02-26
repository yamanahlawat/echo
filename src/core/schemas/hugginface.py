from loguru import logger
from pydantic import BaseModel, Field, SecretStr, model_validator


class HuggingFaceIntegrationSchema(BaseModel):
    """
    Schema for Hugging Face Hub integration. This includes settings for pushing the model
    to the hub, the hub token, and the model ID for the repository.
    """

    push_to_hub: bool = Field(
        description="Whether or not to push the model to the Hugging Face Hub after it's trained.",
    )
    hub_token: SecretStr | None = Field(
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
    hub_organization_id: str | None = Field(
        default=None,
        description="The organization in which to push the trained model to the Hugging Face Hub.",
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
