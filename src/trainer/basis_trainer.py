from datetime import datetime
from logging import Logger
from pathlib import Path
from typing import Any, Callable, Literal

import torch
import torch.nn as nn
import yaml
from omegaconf import DictConfig
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.dataset.utils import inf_loop
from src.logger.base import BaseWriter
from src.logger.utils import plot_function
from src.loss.basis import BasisSetLoss
from src.metrics.base import BaseMetric
from src.metrics.tracker import MetricTracker
from src.model.basis import BasisSet
from src.utils.io import get_root


class BasisTrainer:
    def __init__(
        self,
        cfg: DictConfig,
        device: str,
        dataloaders: dict[str, DataLoader],
        basis: BasisSet,
        criterion: BasisSetLoss,
        metrics: dict[str, list[BaseMetric]],
        optimizer_fn: Callable[[nn.Module], Optimizer],
        lr_scheduler_fn: Callable[[Optimizer], LRScheduler],
        logger: Logger,
        writer: BaseWriter,
        batch_transforms: dict[str, dict[str, nn.Sequential]] | None = None,
        skip_oom: bool = True,
        epoch_len: int | None = None,
    ) -> None:
        self.is_train = True

        self.cfg = cfg
        self.cfg_trainer = self.cfg.trainer

        self.device = device
        self.skip_oom = skip_oom

        self.logger = logger
        self.log_step = self.cfg_trainer.get('log_step', 50)

        self.basis = basis
        self.criterion = criterion
        self.optimizer_fn, self.optimizer = optimizer_fn, None
        self.lr_scheduler_fn, self.lr_scheduler = lr_scheduler_fn, None
        self.batch_transforms = batch_transforms

        # define dataloaders
        if epoch_len is None:
            self.train_dataloader = dataloaders['train']
            self.epoch_len = len(self.train_dataloader)
        else:
            self.train_dataloader = inf_loop(dataloaders['train'])
            self.epoch_len = epoch_len
        self.eval_dataloaders = {k: v for k, v in dataloaders.items() if k != 'train'}

        self.num_functions = self.cfg_trainer.num_functions
        self.current_function_idx = 0
        self.epoch_len = self.cfg_trainer.epoch_len

        # configuration to monitor model performance and save best
        self.save_period = self.cfg_trainer.save_period  # checkpoint each save_period functions

        # setup visualization writer instance
        self.writer = writer

        # define metrics
        self.metrics = metrics
        self.train_metrics_tracker = MetricTracker(
            *self.criterion.loss_names,
            'grad_norm',
            *[m.name for m in self.metrics['train']],
        )
        self.eval_metrics_tracker = MetricTracker(
            *self.criterion.loss_names,
            *[m.name for m in self.metrics['inference']],
        )

        # define checkpoint dir and init everything if required
        self.checkpoint_dir: Path = get_root() / self.cfg_trainer.checkpoint_dir
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)

    def train(self) -> None:
        """
        Wrapper around training process to save model on keyboard interrupt.
        """
        try:
            self._train_process()
        except KeyboardInterrupt as e:
            self._log('Saving model on keyboard interrupt')
            self._save_checkpoint(self.current_function_idx, save_best=False)
            raise e

    def _train_process(self) -> None:
        """
        Full training logic.

        Trains model for multiple epochs, evaluates it on non-train partitions,
        and monitors the performance improvement (for early stopping
        and saving the best checkpoint).
        """
        for function_idx in range(1, self.num_functions + 1):
            self.current_function_idx = function_idx
            self._log(f'{"=" * 80}')
            self._log(f'Training basis function {function_idx}/{self.num_functions}')
            self._log(f'{"=" * 80}')

            function = self.basis.add_function()
            self._log(f'Added new function. Total functions: {len(self.basis)}')
            self._log(f'Model:\n{self.basis}')

            self.optimizer = self.optimizer_fn(function)
            self.lr_scheduler = self.lr_scheduler_fn(self.optimizer)

            result = self._train_function(function_idx=function_idx)
            # save logged information into logs dict
            logs = {'function': function_idx}
            logs.update(result)

            # print logged information to the screen
            self._log(f'\nMetrics:\n{yaml.dump(logs)}')

            if function_idx % self.save_period == 0 or function_idx == self.num_functions:
                self._save_checkpoint(function_idx, save_best=False, only_best=True)

            plot_function(self.cfg.dataset.name, self.basis, self.writer)

    def _train_function(self, function_idx: int) -> dict[str, float]:
        """
        Train a single basis function.

        Parameters
        ----------
        function_idx : int
            Index of the current function being trained.

        Returns
        -------
        logs : dict[str, float]
            Training metrics.
        """
        self.is_train = True
        self.basis.train()
        self.train_metrics_tracker.reset()
        self.writer.set_step((function_idx - 1) * self.epoch_len)
        self.writer.add_scalar('function', function_idx)

        if self.batch_transforms is not None:
            transforms = self.batch_transforms.get('train')
            if transforms is not None:
                for transform_name, transform_fn in transforms.items():
                    if transform_name == 'batch':
                        for item in transform_fn:
                            item.reset()

        pbar = tqdm(
            self.train_dataloader,
            desc=f'Training function {function_idx}/{self.num_functions}',
            total=self.epoch_len,
        )

        for batch_idx, batch in enumerate(pbar):
            try:
                batch = self.process_batch(
                    batch=batch,
                    metric_tracker=self.train_metrics_tracker,
                )
            except torch.cuda.OutOfMemoryError as e:
                if self.skip_oom:
                    self.logger.warning('OOM on batch. Skipping batch.')
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e

            pbar.set_postfix({k: v.item() for k, v in batch.items() if 'loss' in k})
            grad_norm = self._get_grad_norm()
            self.train_metrics_tracker.update('grad_norm', grad_norm)

            # log current results
            if batch_idx > 0 and batch_idx % self.log_step == 0:
                self.writer.set_step(function_idx * self.epoch_len + batch_idx)
                self.writer.add_scalar('lr', self.lr_scheduler.get_last_lr()[0])  #  ty: ignore[possibly-missing-attribute]
                self._log_scalars(self.train_metrics_tracker)
                self._log_batch(batch_idx, batch, function_idx)
                # we don't want to reset train metrics at the start of every epoch
                # because we are interested in recent train metrics
                last_train_metrics = self.train_metrics_tracker.result()
                self.train_metrics_tracker.reset()

            if batch_idx + 1 >= self.epoch_len:
                break

        logs = last_train_metrics
        logs = {f'train_{name}': value for name, value in logs.items()}
        return logs

    def process_batch(
        self,
        batch: dict[str, Any],
        metric_tracker: MetricTracker,
        part: Literal['train', 'val', 'test'] | str = 'train',
    ) -> dict[str, Any]:
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Parameters
        ----------
        batch : dict[str, Any]
            Dict-based batch containing the data from the dataloader.
        metric_tracker : MetricTracker
            MetricTracker object that computes and aggregates the metrics.
            The metrics depend on the type of the partition (train or inference).
        part : {'train', 'val', 'test'}, optional
            Partition type, by default 'train'.

        Returns
        -------
        batch : dict[str, Any]
            Dict-based batch containing the data from the dataloader
            (possibly transformed via batch transform), model outputs, and losses.
        """
        batch = self._to_device(batch)
        batch = self._transform_batch(batch)  # transform batch on device -- faster

        metric_funcs = self.metrics['train' if part == 'train' else 'inference']
        if part == 'train':
            self.optimizer.zero_grad()  # ty: ignore[possibly-missing-attribute]

        output = self.basis(**batch)
        batch.update(output)
        all_losses = self.criterion(basis=self.basis, **batch)
        batch.update(all_losses)

        if part == 'train':
            batch['loss'].backward()  # sum of all losses is always called loss
            self._clip_grad_norm()
            self.optimizer.step()  # ty: ignore[possibly-missing-attribute]
            self.lr_scheduler.step()  # ty: ignore[possibly-missing-attribute]
            self.criterion.update_weights(all_losses)

        # update metrics for each loss (in case of multiple losses)
        for loss_name in self.criterion.loss_names:
            metric_tracker.update(loss_name, batch[loss_name].item())

        for metric in metric_funcs:
            metric_tracker.update(metric.name, metric(**batch))
        return batch

    def _log_batch(
        self,
        batch_idx: int,
        batch: dict[str, Any],
        function_idx: int,
    ) -> None:
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Parameters
        ----------
        batch_idx : int
            Index of the current batch.
        batch : dict[str, Any]
            Dict-based batch after going through the 'process_batch' function.
        function_idx : int
            Current function number.
        """
        self.writer.add_scalar('s_pn', batch['s_pn'])
        self.writer.add_scalar('w_z', batch['w_z'])

    def _to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        """
        Move all necessary tensors to the device.

        Parameters
        ----------
        batch : dict[str, Any]
            Dict-based batch containing the data from the dataloader.

        Returns
        -------
        batch : dict[str, Any]
            Dict-based batch containing the data from the dataloader
            with some of the tensors on the device.
        """
        for key in self.cfg_trainer.device_tensors:
            batch[key] = batch[key].to(self.device)
        return batch

    def _transform_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        """
        Apply batch transforms to tensors.

        Parameters
        ----------
        batch : dict[str, Any]
            Dict-based batch from dataloader.

        Returns
        -------
        batch : dict[str, Any]
            Transformed batch.
        """
        if self.batch_transforms is None:
            return batch

        transform_type = 'train' if self.is_train else 'inference'
        transforms = self.batch_transforms.get(transform_type)

        if transforms is None:
            return batch

        for transform_name, transform_fn in transforms.items():
            if transform_name == 'batch':
                batch = transform_fn(batch)
            elif transform_name in batch:
                batch[transform_name] = transform_fn(batch[transform_name])
        return batch

    def _clip_grad_norm(self) -> None:
        """
        Clips the gradient norm by the value defined in cfg.trainer.max_grad_norm.
        """
        if self.cfg_trainer.get('max_grad_norm', None) is not None:
            clip_grad_norm_(self.basis.parameters(), self.cfg_trainer.max_grad_norm)

    @torch.no_grad()
    def _get_grad_norm(
        self,
        norm_type: float | str | None = 2,
    ) -> float:
        """
        Calculates the gradient norm for logging.

        Parameters
        ----------
        norm_type : float or str or None, optional
            The order of the norm, by default 2.

        Returns
        -------
        total_norm : float
            The calculated norm.
        """
        parameters = self.basis.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        gradients = [p.grad for p in parameters if p.grad is not None]

        if len(gradients) == 0:
            return 0.0

        total_norm = torch.norm(
            torch.stack([torch.norm(grad.detach(), norm_type) for grad in gradients]),
            norm_type,
        )
        return total_norm.item()

    def _log_scalars(self, metric_tracker: MetricTracker) -> None:
        """
        Wrapper around the writer 'add_scalar' to log all metrics.

        Parameters
        ----------
        metric_tracker : MetricTracker
            Calculated metrics.
        """
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f'{metric_name}', metric_tracker[metric_name])

    def _save_checkpoint(
        self,
        function_idx: int,
        save_best: bool = False,
        only_best: bool = False,
    ) -> None:
        """
        Save checkpoint.

        Parameters
        ----------
        function_idx : int
            Current function index.
        save_best : bool, optional
            If True, save as 'model_best.pth', by default False.
        only_best : bool, optional
            If True, save only as best (no duplicate), by default False.
        """
        arch = type(self.basis).__name__
        state = {
            'arch': arch,
            'function_idx': function_idx,
            'num_functions': len(self.basis),
            'state_dict': self.basis.state_dict(),
            'optimizer': self.optimizer.state_dict(),  # ty: ignore[possibly-missing-attribute]
            'lr_scheduler': self.lr_scheduler.state_dict(),  # ty: ignore[possibly-missing-attribute]
            'cfg': self.cfg,
        }

        filename = str(self.checkpoint_dir / f'checkpoint-function-{function_idx}.pth')
        if not (only_best and save_best):
            torch.save(state, filename)
            self._log(f'Saving checkpoint: {filename} ...')
            self.writer.add_checkpoint(filename, str(self.checkpoint_dir.parent))

        if save_best:
            best_path = str(self.checkpoint_dir / 'model_best.pth')
            torch.save(state, best_path)
            self.writer.add_checkpoint(best_path, str(self.checkpoint_dir.parent))
            self._log('Saving current best: model_best.pth ...')

    def _log(
        self,
        message: str,
        message_type: Literal['INFO', 'WARNING', 'DEBUG'] = 'INFO',
    ) -> None:
        """
        Log a message using the configured logger.

        Parameters
        ----------
        message : str
            Message to log.
        message_type : {'INFO', 'WARNING', 'DEBUG'}, optional
            Type of the log message, by default 'INFO'.
        """
        message = f'{type(self).__name__} {message}'
        if self.logger is not None:
            match message_type:
                case 'INFO':
                    self.logger.info(message)
                case 'DEBUG':
                    self.logger.debug(message)
                case 'WARNING':
                    self.logger.warning(message)
                case _:
                    self.logger.info(message)
        else:
            print(f'{datetime.now()} {message_type}: {message}')
