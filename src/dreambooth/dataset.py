from pathlib import Path

from accelerate.logging import get_logger
from torchvision import transforms

logger = get_logger(__name__)


class DreamBoothDataset:
    def __init__(
        self,
        height: int,
        width: int,
        tokenizer,
        instance_data_dir: Path,
        instance_prompt: str,
        num_class_images: int,
        class_data_dir: Path | None = None,
        class_prompt: str | None = None,
        encoder_hidden_states=None,
        class_prompt_encoder_hidden_states=None,
    ):
        logger.info("Creating DreamBooth dataset...")
        self.height = height
        self.width = width
        self.tokenizer = tokenizer

        self.encoder_hidden_states = encoder_hidden_states
        self.class_prompt_encoder_hidden_states = class_prompt_encoder_hidden_states

        self.instance_prompt = instance_prompt
        self.instance_images = list(instance_data_dir.glob("*"))
        self.num_instance_images = len(self.instance_images)
        logger.info(f"Number of instance images: {self.num_instance_images}.")

        self.class_data_dir = class_data_dir
        if class_data_dir:
            self.class_prompt = class_prompt
            self.class_images = list(class_data_dir.glob("*"))
            self.num_class_images = min(len(self.class_images), num_class_images)
            logger.info(f"Using {self.num_class_images} class images for prior preservation loss.")

        self.image_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        if self.num_class_images:
            return max(self.num_class_images, self.num_instance_images)
        else:
            return self.num_instance_images
