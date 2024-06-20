from typing import Literal
from dataclasses import dataclass

from torch import nn


Activation = Literal["relu", "tanh", "sigmoid"]


@dataclass
class DenseConfig:
    hidden_layers: list[int]
    activation: Activation
    dropout: float | None
    batch_norm: bool


def dense_from_hiddens_layers(
    input_size: int,
    output_size: int,
    config: DenseConfig,
) -> nn.Sequential:
    """

    Creates a dense neural network from the input size, hidden layers, output size and activation function.
    Does not include the final activation function.

    """

    layers = []
    in_size = input_size

    for h_size in config.hidden_layers:
        layers.append(nn.Linear(in_size, h_size))
        if config.batch_norm:
            layers.append(nn.BatchNorm1d(h_size))
        layers.append(_get_activation(config.activation))
        if config.dropout is not None:
            layers.append(nn.Dropout(config.dropout))
        in_size = h_size

    layers.append(nn.Linear(in_size, output_size))

    return nn.Sequential(*layers)


def _get_activation(activation: Activation) -> nn.Module:
    """

    Returns the activation function module based on the given string.

    """

    if activation == "relu":
        return nn.ReLU()
    elif activation == "tanh":
        return nn.Tanh()
    elif activation == "sigmoid":
        return nn.Sigmoid()
    else:
        raise ValueError(f"Activation function {activation} not recognized.")
