import logging
from pathlib import Path

import hydra
import torch
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from src.dataset.utils import get_dataloaders
from src.trainer.basis_trainer import BasisTrainer
from src.utils.init import set_random_seed
from src.utils.torch import get_lr_scheduler

logger = logging.getLogger(Path(__file__).name)


@hydra.main(version_base='1.3', config_path='src/configs', config_name='train')
def main(cfg: DictConfig) -> None:
    """
    Main script for training. Instantiates the model, optimizer, scheduler,
    metrics, logger, writer, and dataloaders. Runs Trainer to train and
    evaluate the model.

    Parameters
    cfg : DictConfig
        Hydra experiment config.
    """
    set_random_seed(cfg.trainer.seed)
    logger.info(f'Config:\n{OmegaConf.to_yaml(cfg, resolve=True)}')

    if cfg.trainer.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = cfg.trainer.device
    logger.info(f'Using device: {device}')

    dataloaders, batch_transforms = get_dataloaders(cfg, device)
    basis = instantiate(cfg.model.basis, device=device)
    logger.info(f'Model:\n{basis}')

    def optimizer_fn(model: nn.Module) -> Optimizer:
        return instantiate(
            cfg.optimizer,
            params=[p for p in model.parameters() if p.requires_grad],
        )

    def lr_scheduler_fn(optimizer: Optimizer) -> LRScheduler:
        return get_lr_scheduler(
            cfg=cfg,
            optimizer=optimizer,
            epoch_len=cfg.trainer.epoch_len
            if isinstance(cfg.trainer.get('epoch_len'), int)
            else len(dataloaders['train']),
        )

    criterion = instantiate(cfg.criterion).to(device)
    logger.info(f'Criterion: {criterion}')
    metrics = instantiate(cfg.metrics)
    logger.info(f'Metrics: {metrics}')

    project_config = OmegaConf.to_container(cfg, resolve=True)
    writer = instantiate(cfg.writer, project_config)

    # epoch_len = number of iterations for iteration-based training
    # epoch_len = None or len(dataloader) for epoch-based training
    epoch_len = cfg.trainer.get('epoch_len')
    trainer = BasisTrainer(
        cfg=cfg,
        device=device,
        dataloaders=dataloaders,
        basis=basis,
        criterion=criterion,
        metrics=metrics,
        optimizer_fn=optimizer_fn,
        lr_scheduler_fn=lr_scheduler_fn,
        logger=logger,
        writer=writer,
        skip_oom=cfg.trainer.get('skip_oom', True),
        batch_transforms=batch_transforms,
        epoch_len=epoch_len,
    )

    trainer.train()
    writer.finish()


if __name__ == '__main__':
    main()
