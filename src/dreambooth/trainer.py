import torch
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

    def _check_model_weights(self):
        # Check that all trainable models are in full precision
        low_precision_error_string = (
            "Please make sure to always have all model weights in full float32 precision when starting training - even if"
            " doing mixed precision training. copy of the weights should still be float32."
        )

        if self.accelerator.unwrap_model(self.unet).dtype != torch.float32:
            raise ValueError(
                f"Unet loaded as datatype {self.accelerator.unwrap_model(self.unet).dtype}. {low_precision_error_string}"
            )

        if self.train_text_encoder and self.accelerator.unwrap_model(self.text_encoder).dtype != torch.float32:
            raise ValueError(
                f"Text encoder loaded as datatype {self.accelerator.unwrap_model(self.text_encoder).dtype}. {low_precision_error_string}"
            )

    def setup(self):
        if self.vae:
            self.vae.requires_grad_(False)

        if not self.schema.train_text_encoder:
            self.text_encoder.requires_grad_(False)

        if self.schema.gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()
            if self.schema.train_text_encoder:
                self.text_encoder.gradient_checkpointing_enable()

        self._check_model_weights()

        if self.schema.scale_learning_rate:
            self.schema.learning_rate = (
                self.schema.learning_rate
                * self.schema.gradient_accumulation_steps
                * self.schema.train_batch_size
                * self.accelerator.num_processes
            )

        # parameters_to_optimize = itertools.chain(
        #     self.unet.parameters(),
        #     self.text_encoder.parameters() if self.schema.train_text_encoder else self.unet.parameters(),
        # )

        # optimizer = self._init_optimizer(parameter_to_optimize=parameters_to_optimize)

        # # todo: add pre_compute_text_embeddings
        # pre_computed_encoder_hidden_states = None
        # validation_prompt_encoder_hidden_states = None
        # validation_prompt_negative_prompt_embeds = None
        # pre_computed_class_prompt_encoder_hidden_states = None

    def train(self):
        self.setup()
