<div align="center">
  <h1>💫 Echo</h1>
</div>

Welcome to Echo, the official Python library for training Stable Diffusion models using advanced techniques like Dreambooth.

Our aim is to provide an intuitive and powerful tool for customizing and enhancing Stable Diffusion models, starting with Dreambooth for Stable Diffusion 1.5, and expanding to include support for SDXL models in the future.

## Table of Contents

- [⛩ | Table of Contents](#table-of-contents)
- [👋 | Introduction](#👋--introduction)
- [💻 | Installation](#💻--installation)
- [📈 | Usage](#📈--usage)
- [🗒️ | Requirements](#🗒️--requirements)
- [📅 | Future Plans](#📅--future-plans)
- [📁 | Directory Structure](#📁--directory-structure)
- [🤝 | Community and Contributing](#🤝--community-and-contributing)
- [🪪 | License](#🪪--license)

## 👋 | Introduction

Echo is designed to streamline the process of training and customizing Stable Diffusion models through the Dreambooth technique and beyond. Whether you're an artist, researcher, or enthusiast, Echo provides the tools you need to personalize and enhance your generative models with ease and precision.

## 💻 | Installation

To get started with Echo, clone the repository and install the required dependencies:

### Command Line
```bash
# Clone the repository
git clone https://github.com/yamanahlawat/echo

# Navigate to the Echo directory
cd echo

# Setup the virtual environment and install dependencies
poetry shell
poetry install
```
*Python 3.10 or higher is required to use the latest version of this package.*

### Docker
```
# Clone the repository
git clone https://github.com/yamanahlawat/echo

# Navigate to the Echo directory
cd echo

# Build and run using Docker Compose
docker-compose up --build
```

## 📈 | Usage

### Single GPU Training
#### Initialize the Schema

```python
from src.dreambooth.trainer import DreamboothTrainingSchema
from src.core.constants import (
    LearningRateSchedulerEnum,
    OptimizerEnum,
    SchedulerEnum,
)


schema = DreamboothTrainingSchema(
    # path to .safetensors file or identifier from huggingface.co/models
    pretrained_model_name_or_path="SG161222/Realistic_Vision_V5.1_noVAE",
    # path to optional vae
    pretrained_vae_name_or_path="vae-ft-mse-840000-ema-pruned.safetensors",
    instance_prompt="ohwx", # instance prompt
    instance_data_dir="instance_data_dir", # path to instance images
    width=768,
    height=1024,
    push_to_hub=True,
    hub_token="your-huggingface-hub-token",
    # validation prompts will be used to generate validations images
    # during training for intermediate checkpoints
    validation_prompt="positive validation prompt",
    validation_negative_prompt="negative validation prompt",
    validation_scheduler=SchedulerEnum.EulerAncestralDiscreteScheduler,
    output_dir="output_dir",
    train_text_encoder=True,
    pre_compute_text_embeddings=False,
    class_prompt="man",
    class_data_dir="class_data_dir",
    with_prior_preservation=True,
    mixed_precision="fp16",
    gradient_checkpointing=True,
    set_grads_to_none=True,
    train_batch_size=1,
    optimizer=OptimizerEnum.ADAMW_8BIT,
    weight_decay=1e-2,
    cache_latents=True,
    dataloader_num_workers=8,
    learning_rate_warmup_steps=0,
    learning_rate_scheduler=LearningRateSchedulerEnum.LINEAR,
    learning_rate=2e-6,
    no_half_vae=True,
    # whether to save the model in safetensors format
    save_safetensors=True,
    num_train_epochs=50,
    num_class_images=850,
)
```

There are more fields that the schema supports and there are fields that are optional. For more details about the schema and its fields, see the [Dreambooth Training Schema](./src/dreambooth/schemas/training.py).


#### Initialize the Trainer and start Training

```python
from src.dreambooth.trainer import DreamboothTrainer, DreamboothTrainingSchema

trainer = DreamboothTrainer(schema=schema)
trainer.train()
```

### Multi GPU Training
For training on multiple gpus, just wrap the schema initialization and trainer in `notebook_launcher` from accelerate.

```
from src.core.constants import (
    LearningRateSchedulerEnum,
    OptimizerEnum,
    SchedulerEnum,
)
from src.dreambooth.trainer import DreamboothTrainer, DreamboothTrainingSchema

def train():
    # Initialize the schema like we did above
    schema = DreamboothTrainingSchema(...)
    trainer = DreamboothTrainer(schema=schema)
    trainer.train()


# start training
from accelerate import notebook_launcher
# num_processes is the number of gpus you want to use
notebook_launcher(train, num_processes=4)
```

`num_processes` is the number of gpus you want to use

## 🗒️ | Requirements
`Echo` requires Python 3.10 or higher. All project dependencies are managed with Poetry and can be found in the `pyproject.toml` file.


## 📅 | Future Plans
  - Adding Support for:
    - SDXL models.
    - LoRA.
    - Continuous improvement of the user experience and documentation.

## 📁 | Directory Structure

```BASH
echo/
├── CONTRIBUTING.md
├── README.md
├── docker
│   └── DockerFile
├── docker-compose.yml
├── output
├── poetry.lock
├── pyproject.toml
├── ruff.toml
└── src
    ├── __init__.py
    ├── core
    ├── dreambooth
    ├── scripts
    └── utils
```

## 🤝 | Community and Contributing

We warmly welcome contributions from the community, including bug fixes, new features, and documentation improvements and issues [GitHub](https://github.com/yamanahlawat/echo). If you're interested in contributing, please review our [contributing guide](CONTRIBUTING.md) and submit your pull requests or issues on GitHub.

For questions, support, and discussions, please create a discussion on our [GitHub Discussions page](https://github.com/yamanahlawat/echo/discussions).

## 🪪 | License
This project is licensed under the MIT License.
