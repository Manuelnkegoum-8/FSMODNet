import lightning as L
from torch.optim.lr_scheduler import MultiStepLR,OneCycleLR,StepLR
import numpy as np
import torch, random
import itertools
import torch.nn as nn
from math import ceil
import torch.distributed as dist
from util.misc import get_rank, get_world_size
from .optim_wrapper import OptimWrapper, build_scheduler
from pytorch_lightning.utilities import  rank_zero_info
#from util.utils import WarmupLR
from util import MODELS, METRICS, HOOKS


class Runner(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.model = MODELS.build(config.model)
        

        
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        batch_inputs = (batch['inputs_rgb'], batch['inputs_ir'])
        batch_datasamples = batch['data_samples']
        support_data = batch['support_data']
        loss_dict = self.model(batch_inputs, batch_datasamples, support_data)
        losses = sum(loss_dict[k] for k in loss_dict.keys())
        self.log("Loss/Train", losses,on_step=True,prog_bar=True,logger=True, sync_dist=True)
        self.log_dict(loss_dict,on_step=True,prog_bar=True, logger=True, sync_dist=True)
        return losses

    def configure_optimizers(self):
        optim = OptimWrapper(self.model, self.config.optim_wrapper)
        scheduler = self.config.get('param_scheduler', None)
        optimizer = optim.get_optimizer()
        if scheduler is not None:
            schedulers = build_scheduler(optimizer, scheduler)
            return [optimizer], schedulers
        else:
            return optimizer
    
    
    def validation_step(self, batch, batch_idx):
        batch_inputs = (batch['inputs_rgb'], batch['inputs_ir'])
        batch_datasamples = batch['data_samples']
        batch_datasamples = self.model(batch_inputs, batch_datasamples, None,
                                        self.prototypes, self.support_class_ids)
        self.metric.process(batch_datasamples)

    def on_validation_epoch_start(self):
        self.metric = METRICS.build(self.config.metric)

    def on_validation_epoch_end(self):
        # Gather metrics from all processes
        self.metric.synchronize_between_processes()
        # Compute and log final metrics
        final_metrics = self.metric.compute_metrics()
        self.log_dict(final_metrics, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
    
    def test_step(self, batch, batch_idx):
        batch_inputs = (batch['inputs_rgb'], batch['inputs_ir'])
        batch_datasamples = batch['data_samples']
        batch_datasamples = self.model(batch_inputs, batch_datasamples, None,
                                        self.prototypes, self.support_class_ids)
        # draw predictions and gt boxes for visualization
        self.metric.process(batch_datasamples)
    
    def on_test_epoch_start(self):
        self.metric = METRICS.build(self.config.metric)
    
    def on_test_epoch_end(self):
        # Gather metrics from all processes
        self.metric.synchronize_between_processes()
        # Compute and log final metrics
        final_metrics = self.metric.compute_metrics()
        self.log_dict(final_metrics, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)



class Runner2(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.model = MODELS.build(config.model)
        

        
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        batch_inputs = (batch['inputs_rgb'], batch['inputs_ir'])
        batch_datasamples = batch['data_samples']
        loss_dict = self.model(batch_inputs, batch_datasamples)
        losses = sum(loss_dict[k] for k in loss_dict.keys())
        self.log("Loss/Train", losses,on_step=True,prog_bar=True,logger=True, sync_dist=True)
        self.log_dict(loss_dict,on_step=True,prog_bar=True, logger=True, sync_dist=True)
        return losses

    def configure_optimizers(self):
        optim = OptimWrapper(self.model, self.config.optim_wrapper)
        scheduler = self.config.get('param_scheduler', None)
        optimizer = optim.get_optimizer()
        if scheduler is not None:
            schedulers = build_scheduler(optimizer, scheduler)
            return [optimizer], schedulers
        else:
            return optimizer
    
    
    def validation_step(self, batch, batch_idx):
        batch_inputs = (batch['inputs_rgb'], batch['inputs_ir'])
        batch_datasamples = batch['data_samples']
        batch_datasamples = self.model(batch_inputs, batch_datasamples)
        self.metric.process(batch_datasamples)

    def on_validation_epoch_start(self):
        self.metric = METRICS.build(self.config.metric)

    def on_validation_epoch_end(self):
        # Gather metrics from all processes
        self.metric.synchronize_between_processes()
        # Compute and log final metrics
        final_metrics = self.metric.compute_metrics()
        self.log_dict(final_metrics, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
    
    def test_step(self, batch, batch_idx):
        batch_inputs = (batch['inputs_rgb'], batch['inputs_ir'])
        batch_datasamples = batch['data_samples']
        batch_datasamples = self.model(batch_inputs, batch_datasamples)
        # draw predictions and gt boxes for visualization
        self.metric.process(batch_datasamples)
    
    def on_test_epoch_start(self):
        self.metric = METRICS.build(self.config.metric)
    
    def on_test_epoch_end(self):
        # Gather metrics from all processes
        self.metric.synchronize_between_processes()
        # Compute and log final metrics
        final_metrics = self.metric.compute_metrics()
        self.log_dict(final_metrics, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
    
    def on_load_checkpoint(self, checkpoint):
        state_dict = checkpoint['state_dict']
        for key in list(state_dict.keys()):
            if not key.startswith('model.'):
                new_key = 'model.' + key
                state_dict[new_key] = state_dict.pop(key)
        checkpoint['state_dict'] = state_dict
        
    

    