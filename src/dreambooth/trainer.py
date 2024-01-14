from transformers import AutoTokenizer, PretrainedConfig

from src.core.trainer import BaseTrainer
from src.dreambooth.schemas.training import DreamboothTrainingSchema


class DreamboothTrainer(BaseTrainer):
    def __init__(self, schema: DreamboothTrainingSchema) -> None:
        super().__init__(schema=schema)
        self.logger.info("Initializing Dreambooth trainer...")
        self.tokenizer = self._init_tokenizer()
        self.text_encoder = self._init_text_encoder()

    def _init_tokenizer(self, sub_folder: str = "tokenizer"):
        self.logger.info(f"Initializing tokenizer from pretrained model: {self.schema.pretrained_model_name_or_path}")
        return (
            AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=self.schema.tokenizer_name,
                use_fast=False,
            )
            if self.schema.tokenizer_name
            else AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=self.schema.pretrained_model_name_or_path,
                use_fast=False,
                subfolder=sub_folder,
            )
        )

    def _init_text_encoder(self, sub_folder: str = "text_encoder"):
        text_encoder_config = PretrainedConfig.from_pretrained(
            pretrained_model_name_or_path=self.schema.pretrained_model_name_or_path,
            subfolder=sub_folder,
        )
        model_class_name = text_encoder_config.architectures[0]
        if model_class_name == "CLIPTextModel":
            from transformers import CLIPTextModel

            model_class = CLIPTextModel
        elif model_class_name == "T5EncoderModel":
            from transformers import T5EncoderModel

            model_class = T5EncoderModel
        else:
            self.logger.error(f"Unsupported Text Encoder model class: {model_class_name}")
        self.logger.info(
            f"Initializing text encoder: {model_class_name} from {self.schema.pretrained_model_name_or_path}"
        )
        return model_class.from_pretrained(
            pretrained_model_name_or_path=self.schema.pretrained_model_name_or_path,
            variant=self.schema.variant,
            subfolder=sub_folder,
        )
