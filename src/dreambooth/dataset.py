from pathlib import Path

from accelerate.logging import get_logger

logger = get_logger(__name__)


class DreamBoothDataset:
    def __init__(
        self,
        instance_data_dir: Path,
        instance_prompt: str,
        class_data_dir: Path,
        class_prompt: str,
        num_class_images: int,
        tokenizer,
        height: int,
        width: int,
        encoder_hidden_states,
        class_prompt_encoder_hidden_states,
        tokenizer_max_length: int,
    ):
        pass
