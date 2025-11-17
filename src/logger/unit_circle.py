import io

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from src.model.basis import BasisSet

from .base import BaseWriter


def _get_cos(basis_set: BasisSet, k: int, x: torch.Tensor) -> torch.Tensor:
    basis_set.eval()
    if x.dim() == 1:
        x = x.unsqueeze(1)

    idx = 2 * (k - 1) + 1
    result = basis_set[idx].predict(x)
    return result.squeeze(-1)


def _get_sin(basis_set: BasisSet, k: int, x: torch.Tensor) -> torch.Tensor:
    basis_set.eval()
    if x.dim() == 1:
        x = x.unsqueeze(1)
    idx = 2 * k
    result = basis_set[idx].predict(x)
    return result.squeeze(-1)


def _get_max_k(basis_set: BasisSet) -> int:
    num_functions = len(basis_set) - 1
    return num_functions // 2


def lissajous_from_basis(basis_set: BasisSet, a: int, b: int) -> tuple[torch.Tensor, torch.Tensor]:
    device = basis_set.device
    dtype = basis_set.dtype

    delta, num_points = torch.pi / 2, 2000
    t = torch.linspace(0, 2 * torch.pi, num_points, device=device, dtype=dtype)
    t_shifted = t + delta / a
    points = torch.column_stack([torch.cos(t_shifted), torch.sin(t_shifted)])
    x = _get_cos(basis_set, a, points)
    y = _get_sin(basis_set, b, points)
    return x, y


def fig_to_array(fig: plt.Figure) -> np.ndarray:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    img_array = np.array(img)
    buf.close()
    return img_array


def _plot_lissajous_family(
    basis_set: BasisSet,
    writer: BaseWriter | None = None,
    show: bool = False,
):
    max_k = _get_max_k(basis_set)

    if max_k == 0:
        print('No basis functions for visualization')
        return

    fig, axes = plt.subplots(max_k, max_k, figsize=(3 * max_k, 3 * max_k))

    if max_k == 1:
        axes = [[axes]]
    elif max_k > 1:
        axes = axes.reshape(max_k, max_k)

    for a in range(1, max_k + 1):
        for b in range(1, max_k + 1):
            ax = axes[a - 1][b - 1]
            x, y = lissajous_from_basis(basis_set, a, b)
            ax.plot(x.cpu().numpy(), y.cpu().numpy(), linewidth=1.5)
            ax.set_aspect('equal', 'box')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f'a={a}, b={b}', fontsize=10)
            ax.grid(True, alpha=0.3)

    plt.suptitle('Family of Lissajous Curves', fontsize=16)
    plt.tight_layout()

    if writer is not None:
        img_name = 'lissajous_family'
        img_array = fig_to_array(fig)
        writer.add_image(img_name, img_array)

    if show:
        plt.show()
    else:
        plt.close(fig)


def _plot_basis_functions(
    basis_set: BasisSet,
    writer: BaseWriter | None = None,
    show: bool = False,
):
    max_k = _get_max_k(basis_set)

    if max_k == 0:
        print('No basis functions for visualization (only constant)')
        return

    device = basis_set.device
    dtype = basis_set.dtype

    x = torch.linspace(0, 2 * torch.pi, 2000, device=device, dtype=dtype)
    points = torch.column_stack([torch.cos(x), torch.sin(x)])

    fig, axes = plt.subplots(max_k, 2, figsize=(12, 3 * max_k))

    if max_k == 1:
        axes = axes.reshape(1, 2)

    for k in range(1, max_k + 1):
        ax_cos = axes[k - 1][0]
        y_cos = _get_cos(basis_set, k, points)
        ax_cos.plot(
            x.cpu().numpy(),
            y_cos.cpu().numpy(),
            linewidth=2,
            label=f'Prediction cos({k}x)',
        )
        ax_cos.plot(
            x.cpu().numpy(),
            torch.cos(k * x).cpu().numpy(),
            linewidth=2,
            label=f'Ground Truth cos({k}x)',
        )
        ax_cos.set_title(f'Basis Function: cos({k}x)', fontsize=12)
        ax_cos.set_xlabel('x')
        ax_cos.set_ylabel('y')
        ax_cos.grid(True, alpha=0.3)
        ax_cos.legend()

        ax_sin = axes[k - 1][1]
        y_sin = _get_sin(basis_set, k, points)
        ax_sin.plot(
            x.cpu().numpy(),
            y_sin.cpu().numpy(),
            linewidth=2,
            label=f'Prediction sin({k}x)',
        )
        ax_sin.plot(
            x.cpu().numpy(),
            torch.sin(k * x).cpu().numpy(),
            linewidth=2,
            label=f'Ground Truth sin({k}x)',
        )
        ax_sin.set_title(f'Basis Function: sin({k}x)', fontsize=12)
        ax_sin.set_xlabel('x')
        ax_sin.set_ylabel('y')
        ax_sin.grid(True, alpha=0.3)
        ax_sin.legend()

    plt.tight_layout()
    if writer is not None:
        img_name = 'basis_functions'
        img_array = fig_to_array(fig)
        writer.add_image(img_name, img_array)

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_unit_circle(
    basis: BasisSet,
    writer: BaseWriter | None = None,
    show: bool = False,
):
    _plot_basis_functions(basis, writer, show)
    _plot_lissajous_family(basis, writer, show)
