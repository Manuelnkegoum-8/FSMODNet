import lightning as L
import numpy as np
import torch
from collections import abc
from typing import Dict, List, Union, Type, Optional, Any
import torch.nn.functional as F
from functools import partial
import lightning as L
import torch
from torch.utils.data import DataLoader
from typing import Dict
from functools import partial
import psutil, os
from util import DATASETS, TRANSFORMS

def is_seq_of(seq: Any,
              expected_type: Union[Type, tuple],
              seq_type: Optional[Type] = None) -> bool:
    if seq_type is None:
        exp_seq_type = abc.Sequence
    else:
        assert isinstance(seq_type, type)
        exp_seq_type = seq_type
    if not isinstance(seq, exp_seq_type):
        return False
    for item in seq:
        if not isinstance(item, expected_type):
            return False
    return True

def stack_batch(tensor_list: List[torch.Tensor],
                pad_size_divisor: int = 1,
                pad_value: Union[int, float] = 0) -> torch.Tensor:
    """Stack multiple tensors to form a batch and pad the tensor to the max
    shape use the right bottom padding mode in these images. If
    ``pad_size_divisor > 0``, add padding to ensure the shape of each dim is
    divisible by ``pad_size_divisor``.

    Args:
        tensor_list (List[Tensor]): A list of tensors with the same dim.
        pad_size_divisor (int): If ``pad_size_divisor > 0``, add padding
            to ensure the shape of each dim is divisible by
            ``pad_size_divisor``. This depends on the model, and many
            models need to be divisible by 32. Defaults to 1
        pad_value (int, float): The padding value. Defaults to 0.

    Returns:
       Tensor: The n dim tensor.
    """
    assert isinstance(
        tensor_list,
        list), (f'Expected input type to be list, but got {type(tensor_list)}')
    assert tensor_list, '`tensor_list` could not be an empty list'
    assert len({
        tensor.ndim
        for tensor in tensor_list
    }) == 1, (f'Expected the dimensions of all tensors must be the same, '
              f'but got {[tensor.ndim for tensor in tensor_list]}')

    dim = tensor_list[0].dim()
    num_img = len(tensor_list)
    all_sizes: torch.Tensor = torch.Tensor(
        [tensor.shape for tensor in tensor_list])
    max_sizes = torch.ceil(
        torch.max(all_sizes, dim=0)[0] / pad_size_divisor) * pad_size_divisor
    padded_sizes = max_sizes - all_sizes
    # The first dim normally means channel,  which should not be padded.
    padded_sizes[:, 0] = 0
    if padded_sizes.sum() == 0:
        return torch.stack(tensor_list)
    # `pad` is the second arguments of `F.pad`. If pad is (1, 2, 3, 4),
    # it means that padding the last dim with 1(left) 2(right), padding the
    # penultimate dim to 3(top) 4(bottom). The order of `pad` is opposite of
    # the `padded_sizes`. Therefore, the `padded_sizes` needs to be reversed,
    # and only odd index of pad should be assigned to keep padding "right" and
    # "bottom".
    pad = torch.zeros(num_img, 2 * dim, dtype=torch.int)
    pad[:, 1::2] = padded_sizes[:, range(dim - 1, -1, -1)]
    batch_tensor = []
    for idx, tensor in enumerate(tensor_list):
        batch_tensor.append(
            F.pad(tensor, tuple(pad[idx].tolist()), value=pad_value))
    return torch.stack(batch_tensor)

class DataEngine(L.LightningDataModule):
    def __init__(self, 
                 train_data,
                 val_data=None,
                 test_data=None,
                 support_data=None,
                 mean=[0., 0., 0.],
                 std=[255., 255., 255.],
                 pad_size_divisor=1,
                 pad_value=0,
                 train_bs=16,
                 num_workers=4,
                 val_bs=1,
                 batch_transform=None,
                 debug_memory=False):
        super().__init__()
        self.train_dataset = DATASETS.build(train_data)
        self.val_dataset = DATASETS.build(val_data) if val_data is not None else None
        self.test_dataset = DATASETS.build(test_data) if test_data is not None else None
        self.support_dataset = DATASETS.build(support_data) if support_data is not None else None
        self.mean = mean
        self.std = std
        self.pad_size_divisor = pad_size_divisor
        self.pad_value = pad_value
        self.train_bs = train_bs
        self.val_bs = val_bs
        self.num_workers = num_workers
        self.batch_transforms = list()
        if batch_transform is not None:
            for transform in batch_transform :
                self.batch_transforms.append(TRANSFORMS.build(transform))
        else:
            self.batch_transforms = None

    def normalize(self, img, mean, std):
        mean = torch.tensor(mean).view(-1, 1, 1).to(img.device)
        std = torch.tensor(std).view(-1, 1, 1).to(img.device)
        img = (img - mean) / std
        return img

    def _get_pad_shape(self, _batch_inputs) -> List[tuple]:
        """Get the pad_shape of each image based on data and
        pad_size_divisor."""
        # Process data with `pseudo_collate`.
        if is_seq_of(_batch_inputs, torch.Tensor):
            batch_pad_shape = []
            for ori_input in _batch_inputs:
                pad_h = int(
                    np.ceil(ori_input.shape[1] /
                            self.pad_size_divisor)) * self.pad_size_divisor
                pad_w = int(
                    np.ceil(ori_input.shape[2] /
                            self.pad_size_divisor)) * self.pad_size_divisor
                batch_pad_shape.append((pad_h, pad_w))
        # Process data with `default_collate`.
        elif isinstance(_batch_inputs, torch.Tensor):
            assert _batch_inputs.dim() == 4, (
                'The input of `ImgDataPreprocessor` should be a NCHW tensor '
                'or a list of tensor, but got a tensor with shape: '
                f'{_batch_inputs.shape}')
            pad_h = int(
                np.ceil(_batch_inputs.shape[2] /
                        self.pad_size_divisor)) * self.pad_size_divisor
            pad_w = int(
                np.ceil(_batch_inputs.shape[3] /
                        self.pad_size_divisor)) * self.pad_size_divisor
            batch_pad_shape = [(pad_h, pad_w)] * _batch_inputs.shape[0]
        else:
            pass
        return batch_pad_shape

    def collate_fn(self, batch, batch_transform=None, training=True, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], debug_memory=False):
        
        batch_datas = {}
        inputs_rgb, inputs_ir, data_samples = [], [], []
        support_data = []
        for item in batch:
            rgb_inp = item['inputs_rgb'] # Pil images
            ir_inp = item['inputs_ir']
            support = item.get('support_data', None) # List of support samples for this query
            rgb_inp = torch.from_numpy(np.array(rgb_inp)).float().permute(2, 0, 1) # Convert to tensor and permute to CxHxW
            ir_inp = torch.from_numpy(np.array(ir_inp)).float().permute(2, 0, 1) # Convert to tensor and permute to CxHxW
            rgb_inp = self.normalize(rgb_inp, mean, std)
            ir_inp = self.normalize(ir_inp, mean, std)
            inputs_rgb.append(rgb_inp)
            inputs_ir.append(ir_inp)
            data_samples.append(item['data_samples'])
            if support is None:
                continue
            support_rgb = []
            support_ir = []
            support_targets = []
            for sample in support:
                s_rgb = sample['inputs_rgb']
                s_ir = sample['inputs_ir']
                s_rgb = torch.from_numpy(np.array(s_rgb)).float().permute(2, 0, 1) # Convert to tensor and permute to CxHxW
                s_ir = torch.from_numpy(np.array(s_ir)).float().permute(2, 0, 1) # Convert to tensor and permute to CxHxW
                s_rgb = self.normalize(s_rgb, mean, std)
                s_ir = self.normalize(s_ir, mean, std)
                support_rgb.append(s_rgb)
                support_ir.append(s_ir)
                support_targets.append(sample['data_samples'])
            support_data.append({
                'inputs_rgb': support_rgb, # Stack support samples
                'inputs_ir': support_ir,
                'data_samples': support_targets
            })

        batch_pad_shape = self._get_pad_shape(inputs_rgb)
        batch_datas['inputs_rgb'] = stack_batch(inputs_rgb, 
                                                pad_size_divisor=self.pad_size_divisor,
                                                 pad_value=self.pad_value)
        batch_datas['inputs_ir'] = stack_batch(inputs_ir, 
                                                pad_size_divisor=self.pad_size_divisor,
                                                 pad_value=self.pad_value)
        
        batch_datas['support_data'] = support_data

        batch_input_shape = tuple(inputs_rgb[0].size()[-2:])
        for data_sample, pad_shape in zip(data_samples, batch_pad_shape):
            data_sample['metainfo'].update({
                    'batch_input_shape': batch_input_shape,
                    'pad_shape': pad_shape
                })
        batch_datas['data_samples'] = data_samples

        inputs_rgb = batch_datas['inputs_rgb']
        inputs_ir = batch_datas['inputs_ir']

        if training and batch_transform is not None:
            for transform in batch_transform:
                inputs_rgb, inputs_ir, data_samples = \
                    transform(inputs_rgb, inputs_ir, data_samples)
        
        batch_datas['inputs_rgb'] = inputs_rgb
        batch_datas['inputs_ir'] = inputs_ir
        batch_datas['data_samples'] = data_samples

        return batch_datas

    # ------------------------
    # DataLoader definitions
    # ------------------------
    def train_dataloader(self):
        collate_fn_train = partial(
            self.collate_fn,
            batch_transform=self.batch_transforms,
            training=True,
            debug_memory=True,
            mean=self.mean,
            std=self.std)
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_bs,
            shuffle=True,
            num_workers=self.num_workers,  # cap at 4 workers
            collate_fn=collate_fn_train,
            pin_memory=False,
            persistent_workers=False,
        )

    def val_dataloader(self):
        collate_fn_val = partial(
            self.collate_fn,
            training=False,
            mean=self.mean,
            std=self.std,
        )
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_bs,
            shuffle=False,
            num_workers=2,  # smaller for val
            collate_fn=collate_fn_val,
            pin_memory=False,
            persistent_workers=True,
            drop_last=False
        )

    def test_dataloader(self):
        collate_fn_val = partial(
            self.collate_fn,
            training=False,
            mean=self.mean,
            std=self.std,
        )
        return DataLoader(
            self.test_dataset,
            batch_size=self.val_bs,
            shuffle=False,
            num_workers=2,  # smaller for val
            collate_fn=collate_fn_val,
            pin_memory=False,
            persistent_workers=True,
            drop_last=False
        )