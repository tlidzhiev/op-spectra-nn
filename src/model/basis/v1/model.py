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
        eps: float = 1.0e-3,
        num_iter: int = 1000,
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
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dims[-1], 1, bias=False)
        self.eps: float = eps
        self.num_iter: int = num_iter
        if init_mode is not None:
            self._init_weights(activation=activation, init_mode=init_mode)

    def _predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))

    @torch.no_grad()
    def init_head(self, x: torch.Tensor, y_pred: torch.Tensor) -> None:
        device = x.device
        dtype = x.dtype

        q = self.backbone(x)
        batch_size = x.shape[0]
        qmat = torch.matmul(q.T, y_pred) / batch_size

        smat = torch.matmul(qmat, qmat.T)
        smat += self.eps * torch.eye(smat.shape[0], device=device, dtype=dtype)

        y = torch.randn(smat.shape[0], device=device, dtype=dtype)
        y = y / torch.norm(y, p=2)

        for _ in range(self.num_iter):
            y = torch.linalg.solve(smat, y)
            y = y / torch.norm(y, p=2)

        wmat = y.unsqueeze(0)
        output = torch.matmul(q, wmat.T)
        norm = torch.sqrt(torch.mean(output**2))
        wmat = wmat / norm
        self.head.weight.copy_(wmat)
