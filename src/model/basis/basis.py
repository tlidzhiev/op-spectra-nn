from typing import Any, Iterator, Literal

import torch
import torch.nn as nn

from src.utils.torch import get_dtype

from ..base import BaseModel
from .base_function import BaseFunction
from .utils import get_basis_function


class BasisSet(BaseModel):
    def __init__(
        self,
        model_type: Literal['baseline', 'v1'],
        model_cfg: dict[str, Any],
        device: str = 'cpu',
        dtype: Literal['float32', 'float64'] = 'float32',
    ) -> None:
        super().__init__()
        self.model_type: Literal['baseline'] = model_type
        self.model_cfg: dict[str, Any] = model_cfg
        self.functions = nn.ModuleList()
        self.device: str = device
        self.dtype: torch.dtype = get_dtype(dtype)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        y = [torch.ones(x.shape[0], 1, device=self.device, dtype=self.dtype)]
        for function in self:
            y.append(function.predict(x))
        y = torch.column_stack(y)
        return y

    def forward(self, x: torch.Tensor, z: torch.Tensor, **kwargs) -> dict[str, torch.Tensor]:
        if len(self.functions) <= 0:
            raise RuntimeError('No basis functions are available')
        current_idx = -1
        with torch.enable_grad():
            f_curr = self.functions[current_idx]

            yx, dydx = f_curr(x, return_grad=True)
            yz, dydz = f_curr(z, return_grad=True)

        return {'yx': yx, 'dydx': dydx, 'yz': yz, 'dydz': dydz}

    def add_function(self) -> nn.Module:
        for function in self.functions:
            function.eval()
            for param in function.parameters():
                param.requires_grad_(False)

        new_function = get_basis_function(self.model_type, self.model_cfg)
        new_function = new_function.to(self.device, self.dtype)
        self.functions.append(new_function)
        return new_function

    def train(self) -> None:
        for function in self.functions[:-1]:
            function.eval()
            for param in function.parameters():
                param.requires_grad_(False)

        self.functions[-1].train()
        for param in self.functions[-1].parameters():
            param.requires_grad_(True)

    def eval(self) -> None:
        for function in self.functions:
            function.eval()
            for param in function.parameters():
                param.requires_grad_(False)

    def __getitem__(self, index: int) -> BaseFunction:
        return self.functions[index]

    def __iter__(self) -> Iterator[BaseFunction]:
        return iter(self.functions)

    def __len__(self) -> int:
        return len(self.functions)
