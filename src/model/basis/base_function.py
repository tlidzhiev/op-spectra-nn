from typing import Literal

import torch
import torch.nn as nn

from src.utils.torch import initialize_weights


class BaseFunction(nn.Module):
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self._predict(x)

    def forward(
        self,
        x: torch.Tensor,
        return_grad: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if return_grad:
            y, dydx = self._predict_and_grad(x)
        else:
            y, dydx = self._predict(x), None
        return y, dydx

    def _predict(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(f'{type(self).__name__} must implement _predict method')

    def _predict_and_grad(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_ = x.clone().detach().requires_grad_(True)
        y = self._predict(x_)
        dydx = torch.autograd.grad(
            outputs=y,
            inputs=x_,
            grad_outputs=torch.ones_like(y),
            create_graph=True,
            retain_graph=True,
        )[0]
        return y, dydx

    def _init_weights(
        self,
        activation: Literal['relu', 'leaky_relu', 'silu', 'gelu'],
        init_mode: Literal['normal', 'uniform'],
    ) -> None:
        initialize_weights(self, activation=activation, mode=init_mode)
