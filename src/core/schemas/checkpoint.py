from pathlib import Path

from pydantic import BaseModel, Field


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
