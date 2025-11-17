from typing import Literal

import torch
import torch.nn as nn

from src.utils.torch import get_activation

from ..base_function import BaseFunction


class BasisFunction(BaseFunction):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        activation: Literal['relu', 'leaky_relu', 'silu', 'gelu'] = 'relu',
        init_mode: Literal['normal', 'uniform'] | None = None,
    ) -> None:
        super().__init__()
        _act = get_activation(activation)
        hidden_dims = [input_dim] + hidden_dims
        layers = []
        for i in range(len(hidden_dims) - 1):
            layers.extend(
                [
                    nn.Linear(
                        in_features=hidden_dims[i],
                        out_features=hidden_dims[i + 1],
                        bias=True,
                    ),
                    _act,
                ]
            )
        layers.append(nn.Linear(in_features=hidden_dims[-1], out_features=1, bias=True))

        self.input_dim: int = input_dim
        self.output_dim: int = 1
        self.layers = nn.Sequential(*layers)
        if init_mode is not None:
            self._init_weights(activation=activation, init_mode=init_mode)

    def _predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
