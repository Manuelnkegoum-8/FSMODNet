from .registry import Registry

MODELS = Registry("models")
DATASETS = Registry("datasets")
TRANSFORMS = Registry("transforms")
LOSSES = Registry("losses")
METRICS = Registry("metrics")
HOOKS = Registry("hooks")