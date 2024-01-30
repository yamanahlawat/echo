from bitsandbytes.optim import Adam8bit, Lion, Lion8bit
from loguru import logger
from torch.optim import AdamW
from transformers import Adafactor


def get_adamw_optimizer(
    params, learning_rate: float, betas: tuple[float, float], weight_decay: float, eps: float
) -> AdamW:
    logger.info(
        f"Initializing AdamW optimizer with learning_rate={learning_rate}, betas={betas}, weight_decay={weight_decay}, eps={eps}"
    )
    return AdamW(param=params, lr=learning_rate, betas=betas, weight_decay=weight_decay, eps=eps)


def get_adam8bit_optimizer(
    params, learning_rate: float, betas: tuple[float, float], weight_decay: float, eps: float
) -> Adam8bit:
    logger.info(
        f"Initializing Adam8bit optimizer with learning_rate={learning_rate}, betas={betas}, weight_decay={weight_decay}, eps={eps}"
    )
    return Adam8bit(params=params, lr=learning_rate, betas=betas, weight_decay=weight_decay, eps=eps)


def get_lion_optimizer(params, learning_rate: float, betas: tuple[float, float], weight_decay: float) -> Lion:
    logger.info(
        f"Initializing Lion optimizer with learning_rate={learning_rate}, betas={betas}, weight_decay={weight_decay}"
    )
    return Lion(params=params, lr=learning_rate, betas=betas, weight_decay=weight_decay)


def get_lion8bit_optimizer(params, learning_rate: float, betas: tuple[float, float], weight_decay: float) -> Lion8bit:
    logger.info(
        f"Initializing Lion8bit optimizer with learning_rate={learning_rate}, betas={betas}, weight_decay={weight_decay}"
    )
    return Lion8bit(params=params, lr=learning_rate, betas=betas, weight_decay=weight_decay)


def get_adafactor_optimizer(params, learning_rate: float, weight_decay: float) -> Adafactor:
    logger.info(f"Initializing Adafactor optimizer with learning_rate={learning_rate}, weight_decay={weight_decay}")
    return Adafactor(
        params=params,
        lr=learning_rate,
        clip_threshold=1.0,
        decay_rate=-0.8,
        weight_decay=weight_decay,
        scale_parameter=True,
        relative_step=False,
        warmup_init=False,
    )
