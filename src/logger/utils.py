from typing import Literal

import torch

from src.model.basis import BasisSet

from .base import BaseWriter
from .unit_circle import plot_unit_circle


@torch.no_grad()
def plot_function(
    dataset_name: Literal['unit_circle'],
    basis: BasisSet,
    writer: BaseWriter | None = None,
):
    match dataset_name:
        case 'unit_circle':
            plot_unit_circle(basis, writer, show=False)
        case _:
            raise ValueError(
                f'Unknown dataset name: {dataset_name}. Supported dataset names: "unit_circle"'
            )
