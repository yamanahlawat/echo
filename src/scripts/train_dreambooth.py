import sys

from src.dreambooth.trainer import DreamboothTrainer, DreamboothTrainingSchema


def main(json_data: str):
    # Parse the JSON string into the Pydantic model
    schema = DreamboothTrainingSchema.model_validate_json(json_data)

    trainer = DreamboothTrainer(schema=schema)
    trainer.train()


if __name__ == "__main__":
    json_data = sys.argv[1] if len(sys.argv) > 1 else "{}"
    main(json_data)
