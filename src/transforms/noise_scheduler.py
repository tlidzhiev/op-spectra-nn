import math

import torch
import torch.nn as nn


class NoiseScheduler(nn.Module):
    def __init__(self, dataset_size: int, batch_size: int):
        super().__init__()
        self.dataset_size: int = dataset_size
        self.batch_size: int = batch_size
        self.register_buffer('weight', torch.tensor(0.0))

    def forward(self, batch: dict[str, torch.Tensor | float]) -> dict[str, torch.Tensor | float]:
        if (
            'x' in batch
            and isinstance(batch['x'], torch.Tensor)
            and 'z' in batch
            and isinstance(batch['z'], torch.Tensor)
        ):
            batch['x'], batch['z'], batch['s_pn'] = self._noise(batch['x'], batch['z'])
        return batch

    def _noise(self, x: torch.Tensor, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float]:
        self._update_schedule()
        noise_std = self._get_noise()
        x = torch.normal(x, noise_std)
        z = torch.normal(z, noise_std)
        return x, z, noise_std

    def reset(self) -> None:
        self.weight.fill_(0.0)

    def _update_schedule(self) -> None:
        self.weight = torch.clamp(self.weight + self.batch_size, max=self.dataset_size)

    def _get_noise(self) -> float:
        return 1.0 / math.sqrt(self.weight.item())
