import torch 
import torch.nn as nn 
import torch.nn.functional as F
import cv2
import torchvision
torchvision.disable_beta_transforms_warning()
from util.box_ops import box_xyxy_to_cxcywh
import numpy as np
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F
import torchvision.tv_tensors as tv_tensors
from typing import Any, Callable, cast, Dict, List, Mapping, Optional, Sequence, Type, Union
import collections
from contextlib import suppress
import PIL
import PIL.Image
from numpy import random
from util import TRANSFORMS
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import random as rand_
import importlib.metadata
from torch import Tensor 


Number = Union[int, float]

if importlib.metadata.version('torchvision') == '0.15.2':
    import torchvision
    torchvision.disable_beta_transforms_warning()

    from torchvision.datapoints import BoundingBox as BoundingBoxes
    from torchvision.datapoints import BoundingBoxFormat, Mask, Image, Video
    from torchvision.transforms.v2 import SanitizeBoundingBox as SanitizeBoundingBoxes
    _boxes_keys = ['format', 'spatial_size']

elif '0.17' > importlib.metadata.version('torchvision') >= '0.16':
    import torchvision
    torchvision.disable_beta_transforms_warning()

    from torchvision.transforms.v2 import SanitizeBoundingBoxes
    from torchvision.tv_tensors import (
        BoundingBoxes, BoundingBoxFormat, Mask, Image, Video)
    _boxes_keys = ['format', 'canvas_size']

elif importlib.metadata.version('torchvision') >= '0.17':
    import torchvision
    from torchvision.transforms.v2 import SanitizeBoundingBoxes
    from torchvision.tv_tensors import (
        BoundingBoxes, BoundingBoxFormat, Mask, Image, Video)
    _boxes_keys = ['format', 'canvas_size']

else:
    raise RuntimeError('Please make sure torchvision version >= 0.15.2')

def convert_to_tv_tensor(tensor: Tensor, key: str, box_format='xyxy', spatial_size=None) -> Tensor:
    """
    Args:
        tensor (Tensor): input tensor
        key (str): transform to key

    Return:
        Dict[str, TV_Tensor]
    """
    assert key in ('boxes', 'masks', ), "Only support 'boxes' and 'masks'"
    
    if key == 'boxes':
        box_format = getattr(BoundingBoxFormat, box_format.upper())
        _kwargs = dict(zip(_boxes_keys, [box_format, spatial_size]))
        return BoundingBoxes(tensor, **_kwargs)

    if key == 'masks':
       return Mask(tensor)

#from models.registry import register


"""RandomPhotometricDistort = T.RandomPhotometricDistort
RandomZoomOut = T.RandomZoomOut
RandomHorizontalFlip = T.RandomHorizontalFlip
Resize = T.Resize
ToTensor = T.ToImage
ConvertDtype = T.ToDtype
SanitizeBoundingBox = T.SanitizeBoundingBoxes
RandomCrop = T.RandomCrop"""

__all__ = ['MSPhotoMetricDistortion', 'MSBatchRandomResize', 'Misalign',
           'MSHorizontalFlip', 'MSExpand', 'MSResize', 'MSFixScaleResize',
           'MSRandomIoUCrop', 'FilterAnnotations', 'NormalizeImage']

@TRANSFORMS.register()
class MSPhotoMetricDistortion(object):
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.

    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels

    Required Keys:

    - img (np.uint8)

    Modified Keys:

    - img (np.float32)

    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (sequence): range of contrast.
        saturation_range (sequence): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 brightness_delta: Sequence[Number] = (0.875, 1.125),
                 contrast_range: Sequence[Number] = (0.5, 1.5),
                 saturation_range: Sequence[Number] = (0.5, 1.5),
                 hue_delta: Sequence[Number] = (-0.05, 0.05),
                 prob=0.5) -> None:
        
        self.transform = T.RandomPhotometricDistort(
            brightness=brightness_delta,
            contrast=contrast_range,
            saturation=saturation_range,
            hue=hue_delta,
            p=prob
        )

    def __call__(self, results: dict) -> dict:
        
        transform_rgb, transform_ir = self.transform(results['img_rgb'], results['img_ir'])
        results['img_rgb'] = transform_rgb
        results['img_ir'] = transform_ir
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(brightness_delta={self.brightness_delta}, '
        repr_str += 'contrast_range='
        repr_str += f'{(self.contrast_lower, self.contrast_upper)}, '
        repr_str += 'saturation_range='
        repr_str += f'{(self.saturation_lower, self.saturation_upper)}, '
        repr_str += f'hue_delta={self.hue_delta})'
        return repr_str
    
@TRANSFORMS.register()
class MSHorizontalFlip(object):
    def __init__(self, prob=0.5) -> None:
        self.prob = prob
        self.transform = T.RandomHorizontalFlip(p=prob)

    def __call__(self, results: dict) -> dict:
        
        target = {
                    'boxes': results.get('gt_bboxes', None), 
                    }
        rgb, ir, target = self.transform(results['img_rgb'], results['img_ir'],target)
        results['img_rgb'] = rgb
        results['img_ir'] = ir
        if target is not None:
            results['gt_bboxes'] = target['boxes']
        return results

    def __repr__(self) -> str:
        return self.__class__.__name__ + f'(prob={self.prob})'

@TRANSFORMS.register()
class MSExpand(object):
    """Random expand the image & bboxes & masks & segmentation map.

    Randomly place the original image on a canvas of ``ratio`` x original image
    size filled with mean values. The ratio is in the range of ratio_range.

    Required Keys:

    - img
    - img_shape
    - gt_bboxes (BaseBoxes[torch.float32]) (optional)

    Modified Keys:

    - img
    - img_shape
    - gt_bboxes


    Args:
        mean (sequence): mean value of dataset.
        to_rgb (bool): if need to convert the order of mean to align with RGB.
        ratio_range (sequence)): range of expand ratio.
        seg_ignore_label (int): label of ignore segmentation map.
        prob (float): probability of applying this transformation
    """

    def __init__(self,
                 mean: Sequence[Number] = (0, 0, 0),
                 ratio_range: Sequence[Number] = (1, 4),
                 prob: float = 0.5) -> None:
        self.ratio_range = ratio_range
        self.mean = mean
        self.min_ratio, self.max_ratio = ratio_range
        self.prob = prob
        self.transform = T.RandomZoomOut(fill=self.mean, 
                                         side_range=self.ratio_range, p=prob)
    
    def __call__(self, results: dict) -> dict:
        """Transform function to expand images, bounding boxes, masks,
        segmentation map.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with images, bounding boxes, masks, segmentation
                map expanded.
        """
        target = {
                    'boxes': results.get('gt_bboxes', None),
                    'labels': results.get('gt_bboxes_labels', None),
                    'ignore_flags': results.get('gt_ignore_flags', None)
                    }
        img_rgb, img_ir, target = self.transform(results['img_rgb'], results['img_ir'], target)
        results['img_rgb'] = img_rgb
        results['img_ir'] = img_ir
        if target is not None:
            results['gt_bboxes'] = target['boxes']
            results['gt_bboxes_labels'] = target['labels']
            results['gt_ignore_flags'] = target['ignore_flags']
        results['img_shape'] = img_rgb.size[::-1] # h,w
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, to_rgb={self.to_rgb}, '
        repr_str += f'ratio_range={self.ratio_range}, '
        repr_str += f'prob={self.prob})'
        return repr_str

@TRANSFORMS.register()
class MSRandomIoUCrop(object):
    def __init__(self, prob) -> None:
        #self.min_ious = min_ious
        self.prob = prob
        self.transform = T.RandomIoUCrop( min_scale= 0.3, max_scale= 1, min_aspect_ratio= 0.5, max_aspect_ratio= 2)

    def __call__(self, results: dict) -> dict:

        if random.random() > self.prob:
            return results

        target = {
                    'boxes': results.get('gt_bboxes', None),
                    'labels': results.get('gt_bboxes_labels', None),
                    'ignore_flags': results.get('gt_ignore_flags', None)
                    }
        img_rgb, img_ir, target = self.transform(results['img_rgb'], results['img_ir'], target)
        results['img_rgb'] = img_rgb
        results['img_ir'] = img_ir
        if target is not None:
            results['gt_bboxes'] = target['boxes']
            results['gt_bboxes_labels'] = target['labels']
            results['gt_ignore_flags'] = target['ignore_flags']
        results['img_shape'] = img_rgb.size[::-1] # h,w
        return results

@TRANSFORMS.register()
class SanitizeBoundingBoxX(T.SanitizeBoundingBoxes):
    """[BETA] Remove degenerate/invalid bounding boxes and their corresponding labels and masks.

    .. v2betastatus:: SanitizeBoundingBox transform

    This transform removes bounding boxes and their associated labels/masks that:

    - are below a given ``min_size``: by default this also removes degenerate boxes that have e.g. X2 <= X1.
    - have any coordinate outside of their corresponding image. You may want to
      call :class:`~torchvision.transforms.v2.ClampBoundingBox` first to avoid undesired removals.

    It is recommended to call it at the end of a pipeline, before passing the
    input to the models. It is critical to call this transform if
    :class:`~torchvision.transforms.v2.RandomIoUCrop` was called.
    If you want to be extra careful, you may call it after all transforms that
    may modify bounding boxes but once at the end should be enough in most
    cases.

    Args:
        min_size (float, optional) The size below which bounding boxes are removed. Default is 1.
        labels_getter (callable or str or None, optional): indicates how to identify the labels in the input.
            It can be a str in which case the input is expected to be a dict, and ``labels_getter`` then specifies
            the key whose value corresponds to the labels. It can also be a callable that takes the same input
            as the transform, and returns the labels.
            By default, this will try to find a "labels" key in the input, if
            the input is a dict or it is a tuple whose second element is a dict.
            This heuristic should work well with a lot of datasets, including the built-in torchvision datasets.
    """

    def __init__(self, min_size = 1, min_area = 1):
        labels_getter = lambda x: x[2]['labels']
        super().__init__(min_size, min_area, labels_getter)
    
    def __call__(self, results: dict) -> dict:
        target = {
                    'boxes': results.get('gt_bboxes', None),
                    'labels': torch.from_numpy(results.get('gt_bboxes_labels')),
                    }
        img_rgb, img_ir, target = super().__call__((results['img_rgb'], results['img_ir'], target))
        results['img_rgb'] = img_rgb
        results['img_ir'] = img_ir
        if target is not None:
            results['gt_bboxes'] = target['boxes']
            results['gt_bboxes_labels'] = target['labels']
        return results

@TRANSFORMS.register()
class FilterAnnotations(object):
    def __init__(self, min_wh, keep_empty=True) -> None:
        self.min_w, self.min_h = min_wh
        self.keep_empty = keep_empty

    def __call__(self, results: dict) -> dict:
        gt_bboxes = results.get('gt_bboxes', None)
        gt_labels = results.get('gt_bboxes_labels', None)
        if gt_bboxes is not None and gt_labels is not None:
            widths = gt_bboxes[:, 2] - gt_bboxes[:, 0]
            heights = gt_bboxes[:, 3] - gt_bboxes[:, 1]
            valid_mask = (widths >= self.min_w) & (heights >= self.min_h)
            keep = valid_mask.nonzero(as_tuple=False).squeeze(1)
            if keep.shape[0] == 0:
                if self.keep_empty:
                    return None
            
            keys = ('gt_bboxes', 'gt_bboxes_labels', 'gt_masks', 'gt_ignore_flags')
            for key in keys:
                if key in results:
                    results[key] = tv_tensors.wrap(results[key][keep], like=results[key])

        return results


@TRANSFORMS.register()
class NormalizeImage(object):
    def __init__(self, mean, std) -> None:
       self.mean = mean
       self.std = std

    def cast_to_tv_tensor(self, img):
        #to numpy
        if isinstance(img, PIL.Image.Image):
            img = np.array(img)
        img_tensor = torch.from_numpy(img).float()
        return img_tensor.permute(2, 0, 1) # HWC to CHW
    
    def __call__(self, results: dict) -> dict:
        rgb = self.cast_to_tv_tensor(results['img_rgb'])
        ir = self.cast_to_tv_tensor(results['img_ir'])
        rgb = F.normalize(rgb, mean=self.mean, std=self.std)
        ir = F.normalize(ir, mean=self.mean, std=self.std)
        results['img_rgb'] = rgb
        results['img_ir'] = ir
        return results


@TRANSFORMS.register()
class MSFixScaleResize(object):
    def __init__(self, scale, keep_ratio=False) -> None:
        self.keep_ratio = keep_ratio
        self.scale = scale
    def __call__(self, results: dict) -> dict:
        target = {
                    'boxes': results.get('gt_bboxes', None),
                    'labels': results.get('gt_bboxes_labels', None),
                    'ignore_flags': results.get('gt_ignore_flags', None)
                    }
        img_shape = results['img_rgb'].size # w,h 
        if self.keep_ratio:
            scale_factor = min(self.scale[0] / img_shape[0], self.scale[1] / img_shape[1])
            new_size = (int(img_shape[0] * scale_factor), int(img_shape[1] * scale_factor))
        else:
            new_size = self.scale
        transform = T.Resize(new_size[::-1]) # transform expects (h,w)
        img_rgb, img_ir, target = transform(results['img_rgb'], results['img_ir'], target)
        results['img_rgb'] = img_rgb
        results['img_ir'] = img_ir
        if target is not None:
            results['gt_bboxes'] = target['boxes']
            results['gt_bboxes_labels'] = target['labels']
            results['gt_ignore_flags'] = target['ignore_flags']
        results['img_shape'] = img_rgb.size[::-1] # h,w
        results['scale_factor'] = (img_rgb.size[0] / img_shape[0], img_rgb.size[1] / img_shape[1])
        return results
    

@TRANSFORMS.register()
class MSResize(object):
    def __init__(self, scale, keep_ratio=False) -> None:
        self.keep_ratio = keep_ratio
        self.scale = scale
    def __call__(self, results: dict) -> dict:

        
        img_shape = results['img_rgb'].size # w,h 
        if self.keep_ratio:
            scale_factor = min(self.scale[0] / img_shape[0], self.scale[1] / img_shape[1])
            new_size = (int(img_shape[0] * scale_factor), int(img_shape[1] * scale_factor))
        else:
            new_size = self.scale
        transform = T.Resize(new_size[::-1]) # transform expects (h,w)
        if results.get('gt_bboxes', None) is not None:
            target = {
                    'boxes': results.get('gt_bboxes', None),
                    'labels': results.get('gt_bboxes_labels', None),
                    'ignore_flags': results.get('gt_ignore_flags', None)
                    }
            img_rgb, img_ir, target = transform(results['img_rgb'], results['img_ir'], target)
        else:
            img_rgb, img_ir = transform(results['img_rgb'], results['img_ir'])
            target = None
        results['img_rgb'] = img_rgb
        results['img_ir'] = img_ir
        if target is not None:
            results['gt_bboxes'] = target['boxes']
            results['gt_bboxes_labels'] = target['labels']
            results['gt_ignore_flags'] = target['ignore_flags']
        results['img_shape'] = img_rgb.size[::-1] # h,w
        return results

@TRANSFORMS.register()
class MSBatchRandomResize(object):
    """Batch Random Resize the image & bboxes & masks & segmentation map.

    Required Keys:

    - img
    - img_shape
    - gt_bboxes (BaseBoxes[torch.float32]) (optional)


    Modified Keys:

    - img
    - img_shape
    - scale_factor
    - gt_bboxes


    Args:
        img_scale (list[tuple]): list of image scales for resizing.
        multiscale_mode (str): 'value' or 'range' mode.
            'value': randomly select a scale from img_scale.
            'range': randomly sample a scale from the range.
    """
    def __init__(self, scales = [(800, 800)]) -> None:
        self.img_scales = scales
        
    def __resize(self, rgb, ir, data_sample, new_size) -> dict:
        """Call function to resize images, bounding boxes, masks, and
        segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Resized results, 'img_shape' and 'scale_factor' keys are
                updated in the result dict.
        """ 
        # new_size h,w
        actual_size = data_sample['metainfo']['img_shape'] # H,W need for bboxes
        img_rgb = F.resize(rgb, new_size)
        img_ir = F.resize(ir, new_size)
        if 'bboxes' in data_sample['gt_instances']:
            bboxes = data_sample['gt_instances']['bboxes']
            if bboxes is not None and bboxes.size(0) >0:
                scale_factor = torch.as_tensor([new_size[1] / actual_size[1],
                                                new_size[0] / actual_size[0],
                                                new_size[1] / actual_size[1],
                                                new_size[0] / actual_size[0]], dtype=torch.float32)
                bboxes = bboxes * scale_factor
                data_sample['gt_instances']['bboxes'] = bboxes
        data_sample['metainfo']['img_shape'] = new_size
        data_sample['metainfo']['batch_input_shape'] = new_size
        return img_rgb, img_ir, data_sample

    
    def __call__(self, inputs_rgb, inputs_ir, data_samples) -> dict:
        """Call function to resize images, bounding boxes, masks, and
        segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Resized results, 'img_shape' and 'scale_factor' keys are
                updated in the result dict.
        """
        new_size = random.choice(len(self.img_scales))
        new_size = self.img_scales[new_size]
        Resized_rgb = []
        Resized_ir = []
        Resized_data_samples = []
        for rgb, ir, data_sample in zip(inputs_rgb, inputs_ir, data_samples):
            rgb, ir, data_sample = self.__resize(rgb, ir, data_sample, new_size)
            Resized_rgb.append(rgb)
            Resized_ir.append(ir)
            Resized_data_samples.append(data_sample)
        Resized_rgb = torch.stack(Resized_rgb)
        Resized_ir = torch.stack(Resized_ir)
        return Resized_rgb, Resized_ir, Resized_data_samples
        
    def __repr__(self) -> str:
        return self.__class__.__name__ + f'(img_scales={self.img_scales})'
    

@TRANSFORMS.register()
class Misalign(object):
    """Misalignment augmentation for RGB-IR image pairs.

    shift the RGB and IR images in different directions to generate misaligned image pairs.
    """
    def __init__(self, shift = 10, pad=(0., 0., 0.)) -> None:
        self.shift = shift
        self.pad = pad
        self.horizontal_shift = rand_.sample([-1, 0, 1], 1)[0] * shift
        self.vertical_shift = rand_.sample([-1, 0, 1], 1)[0] * shift
    
    def __call__(self, results: dict) -> dict:
        """The misalignment transform function.

        Args:
            results (dict): The result dict.

        Returns:
            dict: The result dict.
        """
        if self.horizontal_shift == 0 and self.vertical_shift == 0:
            # if there is no shift make a shift to avoid no augmentation
            self.horizontal_shift = rand_.sample([-1, 1], 1)[0] * self.shift
        
        img_rgb = results['img_rgb']
        img_ir = results['img_ir']
        
        w, h = img_rgb.size # PIL Image size is (width, height)
        # shift the IR image
        is_pil = isinstance(img_ir, PIL.Image.Image)
        if is_pil:
            img_ir = np.array(img_ir)
        M = np.float32([[1, 0, self.horizontal_shift], [0, 1, self.vertical_shift]])
        img_ir = cv2.warpAffine(img_ir, M, (w, h), borderValue=self.pad)
        if is_pil:
            img_ir = PIL.Image.fromarray(img_ir)
        results['img_ir'] = img_ir
    
        return results

    def __repr__(self) -> str:

        repr = f'{self.__class__.__name__}(shift={self.shift})'
        repr += f', horizontal_shift={self.horizontal_shift}, vertical_shift={self.vertical_shift}'
        return repr