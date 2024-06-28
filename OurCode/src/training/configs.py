from typing import Literal

from .train import TrainConfig


PAPER_TRAIN_CONFIG = TrainConfig(
    batch_size=32,
    lr=0.0001,
    criterion="bpr",
)

IMPLEMENTATION_TRAIN_CONFIG = TrainConfig(
    batch_size=32,
    lr=0.0001,
    criterion="cross_entropy",
)

TrainConfigType = Literal["paper", "implementation"]


def train_config_from_type(config_type: TrainConfigType) -> TrainConfig:
    if config_type == "paper":
        return PAPER_TRAIN_CONFIG
    elif config_type == "implementation":
        return IMPLEMENTATION_TRAIN_CONFIG
    else:
        raise ValueError(f"Train config {config_type} not supported")
