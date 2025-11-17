from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
import safetensors.torch
import torch
from tqdm.auto import tqdm

from src.dataset.base import BaseDataset
from src.utils.io import get_root, read_json, write_json
from src.utils.torch import get_dtype


class UnitCircleDataset(BaseDataset):
    def __init__(
        self,
        root: Path | str | None = None,
        split: Literal['train', 'test'] = 'train',
        num_samples: int = 100000,
        limit: int | None = None,
        shuffle_index: bool = False,
        instance_transforms: dict[str, Callable] | None = None,
        dtype: Literal['float32', 'float64'] = 'float32',
        force_reindex: bool = False,
        seed: int = 42,
    ):
        self.dtype = get_dtype(dtype)

        if root is None:
            root = get_root() / 'data' / 'circle' / split
        else:
            root = get_root() / root

        index_path = root / 'index.json'
        if index_path.exists() and not force_reindex:
            index: list[dict[str, Any]] = read_json(str(index_path))  # ty: ignore
        else:
            index: list[dict[str, Any]] = self._create_index(num_samples, root, seed)

        super().__init__(
            index=index,
            limit=limit,
            shuffle_index=shuffle_index,
            instance_transforms=instance_transforms,
            use_label=False,
        )

    def _create_index(
        self,
        num_samples: int,
        data_path: Path,
        seed: int,
    ) -> list[dict[str, Any]]:
        np.random.seed(seed)

        index: list[dict[str, Any]] = []
        data_path.mkdir(exist_ok=True, parents=True)

        print(f'Generating {num_samples} circle dataset samples...')
        for i in tqdm(range(num_samples)):
            phi = np.random.uniform(0.0, 2 * np.pi)

            x_coord = np.cos(phi)
            y_coord = np.sin(phi)

            point = np.array([x_coord, y_coord])
            point = torch.from_numpy(point).to(self.dtype)

            save_dict = {'tensor': point}
            save_path = data_path / f'{i:06}.safetensors'
            safetensors.torch.save_file(save_dict, save_path)

            index.append({'path': str(save_path)})

        write_json(index, str(data_path / 'index.json'))

        print(f'Successfully generated {len(index)} circle samples.')
        return index
