import torch
import torch.nn as nn
from torch.optim import Optimizer
import torch.optim.lr_scheduler as torch_schedulers
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.lr_scheduler import SequentialLR
import inspect
import logging
import warnings
from typing import List, Dict, Any, Optional
from typing import Optional, Dict
from lightning.pytorch.utilities.rank_zero import rank_zero_info


logger = logging.getLogger(__name__)

_REGISTRY: dict[str, type] = {
    name: cls
    for name, cls in vars(torch_schedulers).items()
    if isinstance(cls, type) and issubclass(cls, LRScheduler)
}

_REGISTRY_OPTIM: dict[str, type] = {
    name: cls
    for name, cls in vars(torch.optim).items()
    if isinstance(cls, type) and issubclass(cls, Optimizer)
}



def _get_optimizer(type_name: str) -> Optional[type]:
    if type_name not in _REGISTRY_OPTIM:
        raise ValueError(
            f"Unknown optimizer '{type_name}'. "
            f"Available: {sorted(_REGISTRY_OPTIM.keys())}"
        )
    return _REGISTRY_OPTIM[type_name]



class OptimWrapper:
    """
    Builds a PyTorch optimizer from a config.
    Supports optional paramwise_cfg like MMEngine.
    """

    def __init__(self, model: torch.nn.Module, optim_cfg: Dict):
        """
        Args:
            model: LightningModule.model or any torch.nn.Module
            optim_cfg: dict with keys:
                - optimizer: dict with type, lr, weight_decay, etc.
                - paramwise_cfg: optional dict for custom LRs/decays
        """
        self.model = model
        self.optim_cfg = optim_cfg

    @staticmethod
    def build_param_groups(model, base_lr, weight_decay, paramwise_cfg: Optional[Dict]):
        """
        Builds param groups based on paramwise_cfg
        """
        if not paramwise_cfg:
            return model.parameters()

        custom_keys = paramwise_cfg.get("custom_keys", {})
        sorted_keys = sorted(custom_keys.keys(), key=len, reverse=True)
        groups = {}

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            lr = base_lr
            wd = weight_decay

            for key in sorted_keys:
                if key in name:
                    cfg = custom_keys[key]
                    lr = base_lr * cfg.get("lr_mult", 1.0)
                    wd = weight_decay * cfg.get("decay_mult", 1.0)
                    break

            group_key = (lr, wd)
            if group_key not in groups:
                groups[group_key] = {"params": [], "lr": lr, "weight_decay": wd}
            logger.info(f"Param: {name}, lr: {lr}, weight_decay: {wd}")
            groups[group_key]["params"].append(param)

        return list(groups.values())

    def get_optimizer(self) -> torch.optim.Optimizer:
        """Builds the optimizer from the config"""
        opt_cfg = self.optim_cfg["optimizer"].copy()
        opt_type = opt_cfg.pop("type")
        optimizer_cls = _get_optimizer(opt_type)
        if optimizer_cls is None:
            raise ValueError(f"Unknown optimizer type: {opt_type}")

        paramwise_cfg = self.optim_cfg.get("paramwise_cfg", None)

        if paramwise_cfg:
            param_groups = self.build_param_groups(
                self.model,
                base_lr=opt_cfg["lr"],
                weight_decay=opt_cfg.get("weight_decay", 0.0),
                paramwise_cfg=paramwise_cfg
            )
            optimizer = optimizer_cls(param_groups, **opt_cfg)
        else:
            optimizer = optimizer_cls(self.model.parameters(), **opt_cfg)

        return optimizer
    





def _get_scheduler(type_name: str) -> type:
    if type_name not in _REGISTRY:
        raise ValueError(
            f"Unknown scheduler '{type_name}'. "
            f"Available: {sorted(_REGISTRY.keys())}"
        )
    return _REGISTRY[type_name]


def _build_one(optimizer: Optimizer, cfg: dict) -> LRScheduler:
    """Instantiate a single scheduler from a config dict."""
    cfg = dict(cfg)

    type_name = cfg.pop('type')
    cfg.pop('by_epoch',  None)   # Lightning-level key, not a scheduler kwarg
    cfg.pop('begin',     None)   # consumed by build_scheduler for milestones
    cfg.pop('end',       None)   # consumed by build_scheduler for milestones
    cfg.pop('frequency', None)   # Lightning-level key

    cls = _get_scheduler(type_name)

    # Auto-inject total_iters / T_max if the scheduler needs it and it's missing
    sig = inspect.signature(cls.__init__).parameters
    for key in ('total_iters', 'T_max'):
        if key in sig and key not in cfg:
            warnings.warn(
                f"'{type_name}' requires '{key}' but it was not provided in the config. "
                f"Please add it explicitly, e.g. dict(type='{type_name}', {key}=50, ...)."
            )

    return cls(optimizer, **cfg)


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def build_scheduler(
    optimizer: Optimizer,
    param_scheduler: list[dict[str, Any]],
) -> list[dict]:
    """
    Build schedulers from config and return a list of Lightning scheduler dicts.

    Each dict has the form::

        {
            "scheduler": <LRScheduler instance>,
            "interval":  "epoch" | "step",
            "frequency": int,          # default 1
        }

    If all entries share the same ``by_epoch`` value and define ``begin``/``end``
    windows, they are automatically composed into a single ``SequentialLR`` so
    Lightning sees one scheduler with clean milestones.

    Otherwise each entry becomes its own independent Lightning scheduler dict.

    Args:
        optimizer       : The PyTorch optimizer.
        param_scheduler : List of scheduler config dicts (from SLConfig).

    Returns:
        List of Lightning-compatible scheduler dicts, ready to be returned
        directly from ``configure_optimizers``.
    """
    if not param_scheduler:
        warnings.warn('param_scheduler is empty – no LR scheduling will occur.')
        return []

    # ── Try to compose into a single SequentialLR when possible ─────────────
    #    Conditions: all entries have begin/end and share the same by_epoch.

    has_windows   = all('begin' in c and 'end' in c for c in param_scheduler)
    by_epoch_vals = {c.get('by_epoch', True) for c in param_scheduler}
    all_same_unit = len(by_epoch_vals) == 1

    if len(param_scheduler) > 1 and has_windows and all_same_unit:
        by_epoch   = by_epoch_vals.pop()
        interval   = 'epoch' if by_epoch else 'step'
        schedulers = [_build_one(optimizer, c) for c in param_scheduler]

        # milestones = the begin points of every scheduler after the first
        milestones  = [c['begin'] for c in param_scheduler[1:]]

        sequential = SequentialLR(optimizer, schedulers=schedulers, milestones=milestones)
        return [{"scheduler": sequential, "interval": interval, "frequency": 1}]

    # ── Otherwise return one Lightning dict per entry ────────────────────────
    result = []
    for cfg in param_scheduler:
        cfg       = dict(cfg)
        by_epoch  = cfg.get('by_epoch', True)
        frequency = cfg.get('frequency', 1)
        interval  = 'epoch' if by_epoch else 'step'
        scheduler = _build_one(optimizer, cfg)
        result.append({"scheduler": scheduler, "interval": interval, "frequency": frequency})

    return result
