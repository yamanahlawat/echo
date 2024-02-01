from pathlib import Path

from accelerate.logging import get_logger
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms

logger = get_logger(__name__)


class DreamBoothDataset(Dataset):
    def __init__(
        self,
        height: int,
        width: int,
        tokenizer,
        instance_data_dir: Path,
        instance_prompt: str,
        num_class_images: int,
        tokenize_prompt: callable,
        class_data_dir: Path | None = None,
        class_prompt: str | None = None,
        encoder_hidden_states=None,
        class_prompt_encoder_hidden_states=None,
    ):
        logger.info("Creating DreamBooth dataset...")
        self.height = height
        self.width = width
        self.tokenizer = tokenizer
        self.tokenize_prompt = tokenize_prompt

        self.encoder_hidden_states = encoder_hidden_states
        self.class_prompt_encoder_hidden_states = class_prompt_encoder_hidden_states

        self.instance_prompt = instance_prompt
        self.instance_images = list(instance_data_dir.glob("*"))
        self.num_instance_images = len(self.instance_images)
        logger.info(f"Number of instance images: {self.num_instance_images}.")

        self.class_data_dir = class_data_dir
        if self.class_data_dir:
            self.class_prompt = class_prompt
            self.class_images = list(self.class_data_dir.glob("*"))
            self.num_class_images = min(len(self.class_images), num_class_images)
            logger.info(f"Using {self.num_class_images} class images for prior preservation loss.")

        # TODO: add custom transforms
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

    def __getitem__(self, index: int):
        instance_image = Image.open(self.instance_images[index % self.num_instance_images])
        # remove and add it to the transforms
        instance_image = ImageOps.exif_transpose(instance_image)

        if instance_image.mode != "RGB":
            instance_image = instance_image.convert("RGB")

        example = {"instance_images": self.image_transforms(instance_image)}

        if self.encoder_hidden_states:
            example["instance_prompt_ids"] = self.encoder_hidden_states
        else:
            text_inputs = self.tokenize_prompt(prompt=self.instance_prompt)
            example["instance_prompt_ids"] = text_inputs.input_ids
            example["instance_attention_mask"] = text_inputs.attention_mask

        if self.class_data_dir:
            class_image = Image.open(self.class_images[index % self.num_class_images])
            # remove and add it to the transforms
            class_image = ImageOps.exif_transpose(class_image)

            if class_image.mode != "RGB":
                class_image = class_image.convert("RGB")

            example["class_images"] = self.image_transforms(class_image)

            if self.class_prompt_encoder_hidden_states:
                example["class_prompt_ids"] = self.class_prompt_encoder_hidden_states
            else:
                text_inputs = self.tokenize_prompt(prompt=self.class_prompt)
                example["class_prompt_ids"] = text_inputs.input_ids
                example["class_attention_mask"] = text_inputs.attention_mask
        return example


class PromptDataset(Dataset):
    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        return {"prompt": self.prompt, "index": index}
