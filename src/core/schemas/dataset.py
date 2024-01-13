from pydantic import BaseModel, Field


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
