import time
from xml.parsers.expat import model

import torch
import torch.distributed as dist
from torchinfo import summary
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.rank_zero import rank_zero_info
from math import ceil
from util import HOOKS
import os.path as osp
import pandas as pd
import math
import warnings
import os
import numpy as np
from PIL import Image
import cv2
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)
__all__ = ['PrototypeCallback', 'TrainableParamsHook', 'SaveFewShotMetricsCallback', 'LoggingCallback',
           'EMACallback']

@HOOKS.register()
class PrototypeCallback(Callback):

    def __init__(self, episode_size=5, max_iters=100):
        self.episode_size = episode_size
        self.max_iters = max_iters

    def on_test_epoch_start(self, trainer, pl_module):
        if not hasattr(pl_module, 'prototypes'):
            logger.warning("Prototypes not computed. Computing prototypes...")
            self.on_validation_epoch_start(trainer, pl_module)
        return super().on_test_epoch_start(trainer, pl_module)
    @torch.no_grad()
    def on_validation_epoch_start(self, trainer, pl_module):

        support = trainer.datamodule.support_dataset
        device = next(pl_module.model.parameters()).device

        rank = trainer.global_rank
        world_size = trainer.world_size

        logger.info("Computing prototypes...")

        if rank == 0:

            prototypes = []
            proto_cls = []

            for iter_idx, batch in enumerate(support):

                support_rgb = [sample['inputs_rgb'] for sample in batch]
                support_ir = [sample['inputs_ir'] for sample in batch]
                support_targets = [sample['data_samples'] for sample in batch]

                support_class_ids = [
                    sample['gt_instances']['labels']
                    for sample in support_targets
                ]

                num_classes = len(support_class_ids)
                num_episode = ceil(num_classes / self.episode_size)

                episode_proto = []

                for i in range(num_episode):

                    if self.episode_size * (i + 1) <= num_classes:
                        start = self.episode_size * i
                        end = self.episode_size * (i + 1)
                        support_rgb_ = support_rgb[start:end]
                        support_ir_ = support_ir[start:end]
                        support_targets_ = support_targets[start:end]
                        support_class_ids_ = torch.tensor(support_class_ids[start:end]).to(device)
                    else:
                        # take the last episode_size samples
                        support_rgb_ = support_rgb[-self.episode_size:]
                        support_ir_ = support_ir[-self.episode_size:]
                        support_targets_ = support_targets[-self.episode_size:]
                        support_class_ids_ = torch.tensor(support_class_ids[-self.episode_size:]).to(device)
                    
                    support_rgb_ = torch.stack(support_rgb_).to(device)
                    support_ir_ = torch.stack(support_ir_).to(device)

                    prototype = pl_module.model.compute_prototypes(
                        support_rgb_,
                        support_ir_,
                        support_targets_
                    )

                    prototype = [torch.stack(p,dim=0) for p in prototype]
                    episode_proto.append(torch.stack(prototype,dim=0)) # 6 x num_levels x num_support * dim
                    proto_cls.append(support_class_ids_)
                episode_proto = torch.cat(episode_proto,dim=2) # [6 x num_levels x num_cls x dim]
                prototypes.append(episode_proto)


                if iter_idx + 1 >= self.max_iters:
                    break

            # Stack all episodes and average
            prototypes = torch.stack(prototypes, dim=0).mean(dim=0, keepdim=False) 
            support_class_ids = torch.cat(proto_cls, dim=0).to(device)
            payload = [prototypes.cpu(), support_class_ids.cpu()]

        else:
            payload = [None, None]

        # Broadcast to all GPUs
        if dist.is_initialized() and world_size > 1:
            dist.broadcast_object_list(payload, src=0)

        prototypes, support_class_ids = payload
        pl_module.prototypes = prototypes.to(device)
        pl_module.support_class_ids = support_class_ids.to(device)


@HOOKS.register()
class TrainableParamsHook(Callback):

    def __init__(self, Ignore_params=[], stage='meta_learning', metainfo=None):
        self.Ignore_params = Ignore_params
        self.stage = stage
        if metainfo is not None:
            self.novel_classes = metainfo.get('NOVEL_CLASSES', None)
            self.base_classes = metainfo.get('BASE_CLASSES', None)
            self.all_classes = metainfo.get('ALL_CLASSES', None)
            self.cat2id = {cls: idx for idx, cls in enumerate(self.all_classes)}

    def on_fit_start(self, trainer, pl_module):

        if self.stage == 'few_shot_finetune':
            NOVEL_CLASSES = self.novel_classes
            BASE_CLASSES = self.base_classes
            novel_cls_ids = [self.cat2id[cls] for cls in NOVEL_CLASSES]
            base_cls_ids = [self.cat2id[cls] for cls in BASE_CLASSES]
            out_size = len(self.all_classes)
            logger.info(f'Base classes: {BASE_CLASSES}, Novel classes: {NOVEL_CLASSES}')
            for id in range(out_size):
                if id in novel_cls_ids:
                    logger.info(f'\nInitializing classifier for novel class id {id} with normal distribution.')
                    for layer in pl_module.model.meta_score.layers:
                        nn.init.normal_(layer.linear.weight[id])
                    """for cls in pl_module.model.transformer.decoder.class_embed:
                        nn.init.normal_(cls.fc.weight[id])
                        nn.init.constant_(cls.fc.bias[id], -math.log((1 - 0.01) / 0.01))"""
                    

        for name, param in pl_module.named_parameters():
            if any(ignored in name for ignored in self.Ignore_params):
                param.requires_grad = False


@HOOKS.register()
class SaveFewShotMetricsCallback(Callback):
    default_prefix = 'coco'
    def __init__(self, monitor='coco/bbox_mAP_50', rule='greater', summary_file='coco_metrics_summary.csv', save_path='.'):
        self.summary_file = summary_file
        self.monitor = monitor
        self.rule = rule
        self.best_metric = None
        self.save_metrics = None
        self.save_path = save_path

    def on_validation_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics

        if metrics is None or self.monitor not in metrics:
            warnings.warn(f"Monitored metric '{self.monitor}' not found in metrics. Skipping metric saving.")
            return
        
        current_metric = metrics[self.monitor]
        is_better = (
            self.best_metric is None or
            (self.rule == 'greater' and current_metric > self.best_metric) or
            (self.rule == 'less' and current_metric < self.best_metric)
        )

        if is_better:
            self.best_metric = current_metric
            # only save the metrics with default prefix to avoid saving too many metrics from different evaluators
            self.save_metrics = {key: value.item() for key, value in metrics.items() if key.startswith(self.default_prefix)}

    def on_fit_end(self, trainer, pl_module):

        rank = dist.get_rank() if dist.is_initialized() else 0
        # get form trainer.log_dir if it exists, otherwise use self.save_path
        log_dir = getattr(trainer, 'log_dir', None)
        if log_dir is not None:
            # we want to save the summary file in the same directory as the kshot directory, so we need to go 3 level up from log_dir
            save_path = osp.dirname(osp.dirname(osp.dirname(log_dir)))
        else:
            save_path = self.save_path
        summary_file = osp.join(save_path, self.summary_file)
        if rank == 0:
            if osp.exists(summary_file):
                df = pd.read_csv(summary_file)
            else:
                df = pd.DataFrame()
            new_row = pd.DataFrame([self.save_metrics])
            df = pd.concat([df, new_row], ignore_index=True)
            df.to_csv(summary_file, index=False)
            logger.info(f"Saved best metrics to {summary_file}")

def format_time(seconds):
    d, rem = divmod(int(seconds), 86400)
    h, rem = divmod(rem, 3600)
    m, s   = divmod(rem, 60)
    if d > 0:
        eta = f"{d} days {h:02d}:{m:02d}:{s:02d}"  # → "1d 11h 00m 00s"
    else:
        eta = f"{h:02d}:{m:02d}:{s:02d}"  # → "11h 00m 00s"
    return eta


@HOOKS.register()
class LoggingCallback(Callback):
    def __init__(self, log_every_n_steps: int = 100):
        self.log_every_n_steps = log_every_n_steps
        self._start_time: float = 0.0
        self._step_start_time: float = 0.0
        self._data_time: float = 0.0
        self._total_steps: int = 0


    def on_train_start(self, trainer, pl_module) -> None:
        self._start_time = time.time()
        self._total_steps = trainer.estimated_stepping_batches

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx) -> None:
        now = time.time()
        # data_time = wall time between end of last step and start of this one
        self._data_time = now - self._step_start_time if self._step_start_time else 0.0
        self._step_start_time = now

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        

        if (batch_idx+1) % self.log_every_n_steps != 0:
            return

        current_time = time.time()
        step_time = current_time - self._step_start_time
        num_batches = trainer.num_training_batches

        elpased_time = current_time - self._start_time
        steps_done = trainer.global_step
        step_per_sec = steps_done / max(elpased_time, 1e-5)
        remaining_time = (self._total_steps - steps_done) / max(step_per_sec, 1e-5)
        # get eta in format days hh:mm:ss
        eta = format_time(remaining_time)

        memory_usage = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0
        parts = (
            [
                f"Epoch(train) [{trainer.current_epoch + 1}][{batch_idx + 1}/{num_batches}]",
                f"eta: {eta}",
                f"step_time: {step_time:.2f}s",
                f"memory_usage: {memory_usage:.0f} MB",
                f"data_time: {self._data_time:.4f}s",
            ]
            + [f"{k}: {v:.5f}" for k, v in trainer.callback_metrics.items()]
        )

        logger.info("  ".join(parts))

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        self._val_step_start_time = time.time()

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if (batch_idx+1) % self.log_every_n_steps != 0:
            return
        # remainingtime before end of validation epoch
        current_time = time.time()
        elapsed = current_time - self._val_step_start_time
        step_per_sec = (batch_idx + 1) / max(elapsed, 1e-5)
        remaining_time = (trainer.num_val_batches[0] - batch_idx - 1) / max(step_per_sec, 1e-5)
        eta = format_time(remaining_time)

        memory_usage = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0
        parts = (
            [
                f"Epoch(val) [{trainer.current_epoch + 1}][{batch_idx + 1}/{trainer.num_val_batches[0]}]",
                f"eta: {eta}",
                f"memory_usage: {memory_usage:.0f} MB"
            ]

        )
        logger.info("  ".join(parts))
    
    def on_test_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        return self.on_validation_batch_start(trainer, pl_module, batch, batch_idx, dataloader_idx)
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if (batch_idx+1) % self.log_every_n_steps != 0:
            return
        # remainingtime before end of validation epoch
        current_time = time.time()
        elapsed = current_time - self._val_step_start_time
        step_per_sec = (batch_idx + 1) / max(elapsed, 1e-5)
        remaining_time = (trainer.num_test_batches[0] - batch_idx - 1) / max(step_per_sec, 1e-5)
        eta = format_time(remaining_time)

        memory_usage = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0
        parts = (
            [
                f"Epoch(test) [{trainer.current_epoch + 1}][{batch_idx + 1}/{trainer.num_test_batches[0]}]",
                f"eta: {eta}",
                f"memory_usage: {memory_usage:.0f} MB"
            ]

        )
        logger.info("  ".join(parts))



@HOOKS.register()
class EMACallback(Callback):
    def __init__(self, momentum=0.0001, gamma=2000, ema_type='exponential'):
        self.momentum = momentum
        self.ema_state_dict = None
        self.gamma = gamma
        self.type = ema_type
        self.backup_state_dict = None

    def _get_momentum(self, step):
        if self.type == 'linear':
            return self.momentum
        elif self.type == 'exponential':
            return (1 - self.momentum) * math.exp(-step / self.gamma) + self.momentum
        raise ValueError(f"Unsupported EMA type: {self.type}")

    def _get_model_state(self, pl_module):
        """Collect both parameters AND buffers."""
        state = {}
        for name, param in pl_module.model.named_parameters():
            state[name] = param.clone().detach()
        for name, buf in pl_module.model.named_buffers():
            state[name] = buf.clone().detach()
        return state

    def on_train_start(self, trainer, pl_module):
        self.ema_state_dict = self._get_model_state(pl_module)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        momentum = self._get_momentum(trainer.global_step)
        pl_module.log('ema_decay', momentum, prog_bar=True, on_step=True, on_epoch=False)

        with torch.no_grad():
            # update parameters
            for name, param in pl_module.model.named_parameters():
                if name in self.ema_state_dict:
                    self.ema_state_dict[name].mul_(1 - momentum).add_(param, alpha=momentum)
            # update buffers (copy, don't blend — BN stats shouldn't be EMA-blended)
            for name, buf in pl_module.model.named_buffers():
                if name in self.ema_state_dict:
                    self.ema_state_dict[name].copy_(buf)

    def on_validation_epoch_start(self, trainer, pl_module):
        # backup current model state (params + buffers)
        self.backup_state_dict = self._get_model_state(pl_module)
        # swap in EMA state
        with torch.no_grad():
            for name, param in pl_module.model.named_parameters():
                if name in self.ema_state_dict:
                    param.data.copy_(self.ema_state_dict[name])
            for name, buf in pl_module.model.named_buffers():
                if name in self.ema_state_dict:
                    buf.data.copy_(self.ema_state_dict[name])

    def on_validation_epoch_end(self, trainer, pl_module):
        # restore original state
        with torch.no_grad():
            for name, param in pl_module.model.named_parameters():
                if name in self.backup_state_dict:
                    param.data.copy_(self.backup_state_dict[name])
            for name, buf in pl_module.model.named_buffers():
                if name in self.backup_state_dict:
                    buf.data.copy_(self.backup_state_dict[name])
        self.backup_state_dict = None

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        if 'ema_state_dict' in checkpoint:
            pl_module.load_state_dict(checkpoint['ema_state_dict'], strict=True)
            self.ema_state_dict = checkpoint['state_dict']
        else:
            self.ema_state_dict = self._get_model_state(pl_module)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        # Save EMA state dict in checkpoint
        checkpoint['state_dict'] = self.ema_state_dict
        # save original in ema_state
        checkpoint['ema_state_dict'] = checkpoint['state_dict']



@HOOKS.register()
class VizualizationCallback(Callback):
    def __init__(self, interval=1, save_dir='visualization', min_score_threshold=0.25, palette=None):
        self.interval = interval
        self.save_dir = save_dir
        self.min_score_threshold = min_score_threshold
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.palette = palette 

    def draw_boxes(self, img, boxes, labels, scores=None, class_names=None):
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(img)
        boxes = boxes.cpu().tolist()
        labels = labels.cpu().tolist()
        scores = scores.cpu().tolist() if scores is not None else None
        
        for i, (box, label) in enumerate(zip(boxes, labels)):
            x1, y1, x2, y2 = box
            color = self.palette[label]
            
            # draw box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            
            # build label text
            text = class_names[label] if class_names is not None else str(label)
            if scores is not None:
                text += f" {scores[i]:.2f}"
            
            # draw text background + text
            bbox = draw.textbbox((x1, y1), text)
            draw.rectangle(bbox, fill=color)
            draw.text((x1, y1), text, fill=(255, 255, 255))
        
        return img


    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """if (batch_idx + 1) % self.interval != 0:
            return
        
        # visualize the predictions and gt of the first num_samples samples in the batch and save to disk
        # only visualize the first gpu's results to avoid duplicated visualization
        if trainer.global_rank == 0:
            print(batch)"""
        pass
    
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if (batch_idx + 1) % self.interval != 0:
            return
        # visualize the predictions and gt of the first num_samples samples in the batch and save to disk
        # only visualize the first gpu's results to avoid duplicated visualization
        if trainer.global_rank == 0:
            for sample in batch['data_samples']:
                metainfo = sample['metainfo']
                pred_instances = sample['pred_instances']
                img_id = metainfo.get('img_id', None)
                img_path_rgb = metainfo.get('img_path_rgb', None)
                img_path_ir = metainfo.get('img_path_ir', None)
                class_names = metainfo.get('text', None)
                # TODO: visualize the predictions and gt boxes on the images and save to disk
                img_rgb = Image.open(img_path_rgb).convert('RGB')
                img_ir = Image.open(img_path_ir).convert('RGB')
                pred_boxes = pred_instances.get('bboxes', None)
                pred_labels = pred_instances.get('labels', None)
                pred_scores = pred_instances.get('scores', None)
                mask = pred_scores > self.min_score_threshold
                pred_boxes = pred_boxes[mask]
                pred_labels = pred_labels[mask]
                pred_scores = pred_scores[mask]
                img_rgb = self.draw_boxes(img_rgb, pred_boxes, pred_labels, pred_scores, class_names)
                img_ir = self.draw_boxes(img_ir, pred_boxes, pred_labels, pred_scores, class_names)
                # save the visualized images to disk
                # cat the rgb and ir images together for better visualization
                img_cat = Image.new('RGB', (img_rgb.width * 2, img_rgb.height))
                img_cat.paste(img_rgb, (0, 0))
                img_cat.paste(img_ir, (img_rgb.width, 0))
                save_path = os.path.join(self.save_dir, f"{img_id}_pred.png")
                img_cat.save(save_path)
        


    