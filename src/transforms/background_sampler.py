import math

import torch
import torch.nn as nn


class BackgroundSampler(nn.Module):
    def __init__(self, dataset_size: int, batch_size: int):
        super().__init__()
        self.dataloader_size: int = dataset_size // batch_size
        self.batch_size: int = batch_size
        self.register_buffer('weight', torch.tensor(0.0))

    def forward(self, batch: dict[str, torch.Tensor | float]) -> dict[str, torch.Tensor | float]:
        if 'x' in batch and isinstance(batch['x'], torch.Tensor):
            sampled = self.sample(batch['x'])
            batch.update(sampled)
        return batch

    def sample(self, x: torch.Tensor) -> dict[str, torch.Tensor | float]:
        self._update_schedule()
        w_z = self._get_background_weight()
        xmean, xstd = self._compute_stats(x)
        z = xmean + torch.randn_like(x) * xstd
        return {'z': z, 'w_z': w_z}

    def reset(self) -> None:
        self.weight.fill_(0.0)

    def _update_schedule(self) -> None:
        self.weight = torch.clamp(self.weight + self.batch_size, max=self.dataloader_size)

    @staticmethod
    def _compute_stats(x: torch.Tensor, eps: float = 1.0e-10) -> tuple[torch.Tensor, torch.Tensor]:
        xmean = x.mean(dim=0)
        xstd = x.std(dim=0, correction=0) + eps
        return xmean, xstd

    def _get_background_weight(self) -> float:
        return 1.0 / (1.0 + math.sqrt(max(self.weight.item(), 0.0)))
