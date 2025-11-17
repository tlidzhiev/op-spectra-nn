from typing import Literal

import torch
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


def _parse_act(activation: str) -> tuple[str, float | None]:
    """
    Parse activation string to extract name and optional parameter.

    Parameters
    ----------
    activation : str
        Activation string, optionally with parameter (e.g., "leaky_relu:0.2").

    Returns
    -------
    act_name : str
        Activation name in lowercase.
    param : float or None
        Activation parameter if provided, None otherwise.

    Raises
    ------
    ValueError
        If parameter string cannot be converted to float.
    """
    if ':' in activation:
        act_name, param_str = activation.split(':', 1)
        try:
            param = float(param_str)
        except ValueError:
            raise ValueError(
                f'Invalid activation parameter "{param_str}" in "{activation}". '
                f'Parameter must be a number.'
            )
        return act_name.lower(), param
    return activation.lower(), None


def get_activation(activation: str) -> nn.Module:
    """
    Get activation module by name.

    Parameters
    ----------
    activation : str
        Activation name, optionally with parameter (e.g., "leaky_relu:0.2").
        Supported: "relu", "leaky_relu", "gelu", "silu".

    Returns
    -------
    nn.Module
        PyTorch activation module.

    Raises
    ------
    ValueError
        If activation type is not supported.
    """
    act_name, param = _parse_act(activation)
    match act_name:
        case 'relu':
            return nn.ReLU()
        case 'leaky_relu':
            slope = param if param is not None else 0.01
            return nn.LeakyReLU(slope)
        case 'gelu':
            return nn.GELU()
        case 'silu':
            return nn.SiLU()
        case _:
            raise ValueError(
                f'Unknown activation type: "{act_name}". '
                f'Supported types: "relu", "leaky_relu", "gelu", "silu"'
            )


def initialize_weights(
    module: nn.Module,
    activation: str,
    mode: Literal['normal', 'uniform'],
) -> None:
    """
    Initialize weights for neural network module using Kaiming initialization.

    Parameters
    ----------
    module : nn.Module
        PyTorch module to initialize.
    activation : str
        Activation function name.
        Supported: "relu", "leaky_relu", "silu", "gelu".
    mode : {'normal', 'uniform'}
        Initialization mode.

    Raises
    ------
    ValueError
        If mode or activation type is not supported.
    """
    if mode not in ['normal', 'uniform']:
        raise ValueError(
            f'Unknown initialization mode: "{mode}". Supported modes: "normal", "uniform".'
        )

    act_name, param = _parse_act(activation)
    param = param if param is not None else 0.0

    if act_name not in ['relu', 'leaky_relu', 'silu', 'gelu']:
        raise ValueError(
            f"Unknown activation type for initialization: '{act_name}'. "
            f'Supported types: {", ".join(["relu", "leaky_relu", "silu", "gelu"])}'
        )
    nonlinearity = 'leaky_relu' if act_name == 'leaky_relu' else 'relu'
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            init_fn = nn.init.kaiming_normal_ if mode == 'normal' else nn.init.kaiming_uniform_
            init_fn(m.weight, a=param, nonlinearity=nonlinearity)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def get_lr_scheduler(
    cfg: DictConfig,
    optimizer: Optimizer,
    epoch_len: int,
) -> LRScheduler:
    """
    Get learning rate scheduler from config.

    Parameters
    ----------
    cfg : DictConfig
        Configuration object.
    optimizer : Optimizer
        PyTorch optimizer.
    epoch_len : int
        Number of steps in each epoch.

    Returns
    -------
    LRScheduler
        Learning rate scheduler.
    """
    if cfg.lr_scheduler.scheduler.name == 'constant':
        num_training_steps, num_warmup_steps = None, None
    else:
        num_training_steps = cfg.trainer.num_epochs * epoch_len
        num_warmup_steps = int(
            round(num_training_steps * cfg.lr_scheduler.get('warmup_ratio', 0.03))
        )
    return instantiate(
        cfg.lr_scheduler.scheduler,
        optimizer=optimizer,
        num_training_steps=num_training_steps,
        num_warmup_steps=num_warmup_steps,
    )


def get_dtype(dtype: Literal['float32', 'float64'] = 'float32') -> torch.dtype:
    """
    Get PyTorch dtype from string.

    Parameters
    ----------
    dtype : {'float32', 'float64'}, optional
        Data type name, by default 'float32'.

    Returns
    -------
    torch.dtype
        PyTorch data type.

    Raises
    ------
    ValueError
        If dtype is not supported.
    """
    match dtype:
        case 'float32':
            return torch.float32
        case 'float64':
            return torch.float64
        case _:
            raise ValueError(f'Unknown dtype: {dtype}. Supported dtypes: "float32" or "float64"')
