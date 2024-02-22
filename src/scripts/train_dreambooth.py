import json
import sys

from src.dreambooth.trainer import DreamboothTrainer, DreamboothTrainingSchema


def main(config_file_path: str):
    # Load the JSON configuration from the file
    with open(config_file_path, "r") as json_file:
        dreambooth_config = json.load(json_file)

    # Parse the JSON data into the Pydantic model
    schema = DreamboothTrainingSchema.model_validate(dreambooth_config)

    trainer = DreamboothTrainer(schema=schema)
    trainer.train()


if __name__ == "__main__":
    config_file_path = sys.argv[1] if len(sys.argv) > 1 else "/workspace/default_config.json"
    main(config_file_path)
