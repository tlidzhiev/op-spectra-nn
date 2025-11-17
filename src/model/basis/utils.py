from typing import Any, Literal

from .base_function import BaseFunction
from .baseline import BasisFunction as BaselineBasisFunction


def get_basis_function(
    model_type: Literal['baseline', 'v1'],
    model_cfg: dict[str, Any],
) -> BaseFunction:
    match model_type:
        case 'baseline':
            return BaselineBasisFunction(**model_cfg)
        case _:
            raise ValueError(f'Unknown model type: {model_type}. Supported model types: "baseline"')
