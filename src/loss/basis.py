import math
from typing import Literal

import torch

from src.model.basis import BasisSet

from .base import BaseLoss


class BasisSetLoss(BaseLoss):
    def __init__(
        self,
        w_orth: float = 1.0,
        w_norm: float = 1.0,
        w_grad: float = 1.0,
        weight_update_strategy: Literal['fixed', 'adaptive'] = 'fixed',
        kappa: float | None = None,
    ):
        if weight_update_strategy != 'fixed':
            if kappa is None:
                raise ValueError('kappa must be specified for adaptive weight update strategy')

        super().__init__()
        self.w_orth: float = w_orth
        self.w_norm: float = w_norm
        self.w_grad: float = w_grad
        self.weight_update_strategy: Literal['fixed', 'adaptive'] = weight_update_strategy
        self.kappa: float | None = kappa
        self.loss_names: list[str] = ['loss', 'loss_orth', 'loss_norm', 'loss_grad']
        self.loss_weight_names: list[str] = ['w_orth', 'w_norm', 'w_grad']

    def forward(
        self,
        basis: BasisSet,
        x: torch.Tensor,
        yx: torch.Tensor,
        dydx: torch.Tensor,
        z: torch.Tensor,
        yz: torch.Tensor,
        dydz: torch.Tensor,
        w_z: float,
        **kwargs,
    ) -> dict[str, torch.Tensor | float]:
        variance = self._variance(yx, yz, w_z)
        loss_norm = self._loss_norm(variance)
        loss_grad = self._loss_grad(dydx, dydz, w_z, variance)
        loss_orth = self._loss_orth(
            basis_set=basis,
            x=x,
            z=z,
            yx=yx,
            yz=yz,
            w_z=w_z,
            variance=variance,
        )
        total_loss = self.w_orth * loss_orth + self.w_norm * loss_norm + self.w_grad * loss_grad

        return {
            'loss': total_loss,
            'loss_orth': loss_orth,
            'w_orth': self.w_orth,
            'loss_norm': loss_norm,
            'w_norm': self.w_norm,
            'loss_grad': loss_grad,
            'w_grad': self.w_grad,
        }

    @staticmethod
    def _variance(yx: torch.Tensor, yz: torch.Tensor, w_z: float) -> torch.Tensor:
        var_x = torch.var(yx, correction=0)
        var_z = torch.var(yz, correction=0)
        variance = torch.lerp(var_x, var_z, w_z)
        return variance

    @staticmethod
    def _loss_norm(variance: torch.Tensor) -> torch.Tensor:
        return ((variance + 1.0) / (2.0 * torch.sqrt(variance))) - 1.0

    @staticmethod
    def _loss_grad(
        dydx: torch.Tensor,
        dydz: torch.Tensor,
        w_z: float,
        variance: torch.Tensor,
    ) -> torch.Tensor:
        grad_norm_x = torch.mean(dydx.square())
        grad_norm_z = torch.mean(dydz.square())
        grad_norm = torch.lerp(grad_norm_x, grad_norm_z, w_z)
        return grad_norm / variance

    def _loss_orth(
        self,
        basis_set: BasisSet,
        x: torch.Tensor,
        z: torch.Tensor,
        yx: torch.Tensor,
        yz: torch.Tensor,
        w_z: float,
        variance: torch.Tensor,
    ) -> torch.Tensor:
        penalty = torch.lerp(torch.mean(yx).square(), torch.mean(yz).square(), w_z)

        current_abs_idx = len(basis_set) - 1
        for k, f_old in enumerate(basis_set):
            if k == current_abs_idx:
                continue

            with torch.no_grad():
                y_old_x: torch.Tensor = f_old(x, return_grad=False)[0]
                y_old_z: torch.Tensor = f_old(z, return_grad=False)[0]

                norm_sq_x = torch.mean(y_old_x.square())
                norm_sq_z = torch.mean(y_old_z.square())
                norm_sq = torch.lerp(norm_sq_x, norm_sq_z, w_z)
                norm = torch.sqrt(norm_sq)

                y_old_x = y_old_x / norm
                y_old_z = y_old_z / norm

                proj_x = torch.mean(yx * y_old_x)
                proj_z = torch.mean(yz * y_old_z)
                proj = torch.lerp(proj_x, proj_z, w_z)

            penalty += proj.square()
        return penalty / variance

    def update_weights(self, loss_output: dict[str, torch.Tensor | float]):
        if self.weight_update_strategy == 'adaptive':
            self._update_weights_adaptive(loss_output)
        elif self.weight_update_strategy == 'fixed':
            pass  # Weights remain constant
        else:
            raise ValueError(
                f'Unknown weight update strategy: "{self.weight_update_strategy}". '
                f'Expected "fixed" or "adaptive".'
            )

    def _update_weights_adaptive(self, loss_output: dict[str, torch.Tensor | float]):
        lo = loss_output['loss_orth'].item()  # ty: ignore[possibly-missing-attribute]
        ln = loss_output['loss_norm'].item()  # ty: ignore[possibly-missing-attribute]

        w_orth = 1.0
        w_norm = math.exp(-self.kappa * lo)  # ty: ignore[unsupported-operator]
        w_grad = math.exp(-self.kappa * max(lo, ln))  # ty: ignore[unsupported-operator]

        s = 1.0
        self.w_orth: float = w_orth / s
        self.w_norm: float = w_norm / s
        self.w_grad: float = w_grad / s
