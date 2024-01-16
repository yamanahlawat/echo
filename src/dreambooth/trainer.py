from diffusers import UNet2DConditionModel
from transformers import AutoTokenizer, PretrainedConfig

from src.core.trainer import BaseTrainer
from src.dreambooth.schemas.training import DreamboothTrainingSchema


class DreamboothTrainer(BaseTrainer):
    def __init__(self, schema: DreamboothTrainingSchema) -> None:
        super().__init__(schema=schema)
        self.logger.info("Initializing Dreambooth trainer...")
        self.tokenizer = self._init_tokenizer()

        # text encoder
        self.text_encoder_model_class = self._get_text_encoder_model_class()
        self.text_encoder = self._init_text_encoder()

        # register hooks for saving and loading
        self.accelerator.register_save_state_pre_hook(self._save_model_hook)
        self.accelerator.register_load_state_pre_hook(self._load_model_hook)

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

    def _get_text_encoder_model_class(self, sub_folder: str = "text_encoder"):
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
        return model_class

    def _init_text_encoder(self, sub_folder: str = "text_encoder"):
        self.logger.info(
            f"Initializing text encoder: {self.text_encoder_model_class} from {self.schema.pretrained_model_name_or_path}"
        )
        return self.text_encoder_model_class.from_pretrained(
            pretrained_model_name_or_path=self.schema.pretrained_model_name_or_path,
            variant=self.schema.variant,
            subfolder=sub_folder,
        )

    def _save_model_hook(self, models, weights, output_dir):
        if self.accelerator.is_main_process:
            for model in models:
                sub_dir = (
                    "unet" if isinstance(model, type(self.accelerator.unwrap_model(self.unet))) else "text_encoder"
                )
                model.save_pretrained(output_dir / sub_dir)
                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

    def _load_model_hook(self, models, input_dir):
        while len(models) > 0:
            # pop models so that they are not loaded again
            model = models.pop()

            if isinstance(model, type(self.accelerator.unwrap_model(self.text_encoder))):
                # load transformers style into model
                load_model = self.text_encoder_model_class.from_pretrained(
                    pretrained_model_name_or_path=input_dir, subfolder="text_encoder"
                )
                model.config = load_model.config
            else:
                load_model = UNet2DConditionModel.from_pretrained(
                    pretrained_model_name_or_path=input_dir, subfolder="unet"
                )
                model.register_to_config(**load_model.config)

            model.load_state_dict(load_model.state_dict())
            del load_model
