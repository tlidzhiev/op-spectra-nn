from typing import Any

import torch


class collate_fn:
    """
    Collate function for datasets.

    Converts individual dataset items into batches.
    """

    def __init__(self, use_label: bool) -> None:
        """
        Parameters
        ----------
        use_label : bool
            Whether to include labels in the batch.
        """
        self.use_label = use_label

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """
        Collate and pad fields in the dataset items.

        Converts individual items into a batch.

        Parameters
        ----------
        batch : list[dict[str, Any]]
            List of objects from dataset.__getitem__.

        Returns
        -------
        dict[str, torch.Tensor]
            Dict containing batch-version of the tensors.
        """
        x_batch: list[torch.Tensor]
        labels_batch: list[int]

        x_batch, labels_batch = [], []
        for item in batch:
            x_batch.append(item['x'])

            if self.use_label:
                labels_batch.append(item['c'])

        x = torch.stack(x_batch)
        labels = torch.tensor(labels_batch, dtype=torch.long) if self.use_label else None

        result_batch = {'x': x}
        if self.use_label:
            result_batch.update({'labels': labels})
        return result_batch
