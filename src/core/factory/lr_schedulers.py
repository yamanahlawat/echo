from loguru import logger
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts


def get_cosine_annealing_warm_restarts_scheduler(
    optimizer: Optimizer, T_0: int, T_mult: int = 1, eta_min: float = 1e-6, verbose: bool = False
):
    logger.info(f"Initializing CosineAnnealingWarmRestarts scheduler: T_0={T_0}, T_mult={T_mult}, eta_min={eta_min}")
    return CosineAnnealingWarmRestarts(
        optimizer=optimizer, T_0=T_0, T_mult=1, eta_min=1e-6, last_epoch=-1, verbose=verbose
    )


def get_cosine_annealing_scheduler(optimizer: Optimizer, T_max: int, eta_min: float = 1e-6, verbose: bool = False):
    logger.info(f"Initializing CosineAnnealingLR scheduler: T_max={T_max}, eta_min={eta_min}")
    return CosineAnnealingLR(optimizer=optimizer, T_max=T_max, eta_min=1e-6, last_epoch=-1, verbose=verbose)
