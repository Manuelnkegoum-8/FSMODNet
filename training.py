# === Standard Libraries ===
import os
import json
import random
import argparse
import sys
import numpy as np
import logging
# === PyTorch & Lightning ===
import torch
import torch.nn as nn
import lightning as L
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    ModelSummary,
    Timer,
)
from lightning.pytorch.utilities import rank_zero_info
from lightning.pytorch.loggers import TensorBoardLogger
from util.slconfig import SLConfig, DictAction
# === Project Modules ===
from engine import Runner, DataEngine
from fsmodnet import *
from datasets import *
import warnings
from util import HOOKS
warnings.filterwarnings("ignore", message=".*__flops__ or __params__ are already defined.*")

import logging
from datetime import datetime


"""def setup_logging(output_dir: str, level: int = logging.INFO) -> None:
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(output_dir, f"train_{timestamp}.log")),
            logging.StreamHandler(sys.stdout),
        ],
    )"""
def setup_logging(output_dir: str, level: int = logging.INFO) -> None:
    # only rank 0 logs — all other ranks are silent
    rank = int(os.environ.get("LOCAL_RANK", 0))
    if rank != 0:
        logging.disable(logging.CRITICAL)
        return

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(level)

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    file_handler = logging.FileHandler(os.path.join(output_dir, f"train_{timestamp}.log"))
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)


# ========================
# Argument Parser
# ========================
parser = argparse.ArgumentParser(description='FSMODNet')

# Data args
parser.add_argument('--output_dir', default='results', type=str)
parser.add_argument('--config', default='myconf.py', type=str)
parser.add_argument('--checkpoint', default=None, type=str)
parser.add_argument('--test_only', action='store_true')
parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
# Optimization args
parser.add_argument('--amp', action='store_true')
parser.add_argument('--gpu', default=4, type=int)
parser.add_argument('--resume', action='store_true')

# ========================
# Main Launcher
# ========================
def launch(args):
    L.seed_everything(42) # Set seed for reproducibility

    os.makedirs(args.output_dir, exist_ok=True)
    assert torch.cuda.is_available(), "CUDA is not available"
    args.device = 'cuda'

    cfg = SLConfig.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    
    cfg.dump(os.path.join(args.output_dir, "config_cfg.py"))

    # === DataModule ===
    train_module = DataEngine(
        train_data=cfg.train_data,
        val_data=cfg.val_data,
        test_data=cfg.test_data,
        support_data=cfg.support_data,
        mean=cfg.mean,
        std=cfg.std,
        train_bs=cfg.train_bs,
        val_bs=cfg.val_bs,
        num_workers=cfg.num_workers,
        batch_transform=cfg.batch_transform,
    )

    # === Model ===
    output_dir = args.output_dir
    setup_logging(output_dir)
    if args.checkpoint is not None:
        checkpoint_path = args.checkpoint
        #logger.info(f"Loading model from checkpoint: {checkpoint_path}")
        model = Runner.load_from_checkpoint(checkpoint_path, config=cfg)
    else:
        model = Runner(cfg)

    # === Callbacks ===
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        filename='detector',
        monitor=cfg.monitor_metric,
        save_top_k=1,
        mode='max',
        save_weights_only=False,
    )
    tenso_logger = TensorBoardLogger(save_dir=output_dir, name='logs', max_queue=100,
        flush_secs=10)


    default_callbacks = [
        RichModelSummary(max_depth=3),
        LearningRateMonitor(logging_interval='step'),
        checkpoint_callback,
        Timer(),
        ]
    callbacks = list()
    custom_callbacks = cfg.get('custom_callbacks', [])
    for custom_callback in custom_callbacks:
        # modify the save path for the custom callback if it has a 'save_path' attribute
        if custom_callback.get('save_dir', False):
            custom_callback['save_dir'] = os.path.join(output_dir, custom_callback['save_dir'])
        callback = HOOKS.build(custom_callback)
        callbacks.append(callback)
    callbacks.extend(default_callbacks)


    num_devices = torch.cuda.device_count()
    # === Trainer ===
    trainer = L.Trainer(
        sync_batchnorm=True,
        max_epochs=cfg.epochs,
        logger=[tenso_logger],
        num_sanity_val_steps=0,
        #plugins=[NonAtomicCheckpointIO()],
        callbacks=callbacks,
        log_every_n_steps=cfg.freq,
        check_val_every_n_epoch=cfg.save_checkpoint_interval,
        precision="16-mixed" if args.amp else "32",
        gradient_clip_val=cfg.clip_max_norm,
        gradient_clip_algorithm="norm",
        devices=num_devices,
        enable_progress_bar=False,
        #strategy='ddp_find_unused_parameters_true'
    )
    logger = logging.getLogger(__name__)

    if args.test_only:
        trainer.test(model=model, datamodule=train_module)
        return
    
    resume = args.resume
    if resume:
        last_checkpoint = os.path.join(output_dir, 'detector.ckpt')
        if os.path.exists(last_checkpoint):
            logger.info(f"Resuming from checkpoint: {last_checkpoint}")
            trainer.fit(model=model, datamodule=train_module, ckpt_path=last_checkpoint)
        else:
            logger.info("No checkpoint found. Starting training from scratch.")
            trainer.fit(model=model, datamodule=train_module)
    else:
        # === Train ===
        trainer.fit(model=model, datamodule=train_module)




if __name__ == '__main__':
    args = parser.parse_args()
    launch(args)
