import numpy as np
import warnings
import os.path as osp
from PIL import Image
from copy import deepcopy
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union
import numpy as np
import pycocotools.mask as maskUtils
import torch
import importlib
import torchvision.transforms.v2 as T
import random
from .utils import convert_to_tv_tensor, to_tensor
from util import TRANSFORMS



__all__ = ['LoadMSImagesFromFile', 'LoadAnnotations', 'MSPackDetInputs', 'RandomLoadText']


@TRANSFORMS.register()
class LoadMSImagesFromFile(object):
    """Load an image from file.

    Required Keys:

    - img_path

    Modified Keys:

    - img
    - img_shape
    - ori_shape

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        
        ignore_empty (bool): Whether to allow loading empty image or file path
            not existent. Defaults to False.
    """

    def __init__(self,
                 to_float32: bool = False,
                 ignore_empty: bool = False,
                 backend_args: Optional[dict] = None) -> None:
        self.ignore_empty = ignore_empty
        self.to_float32 = to_float32
        self.file_client_args: Optional[dict] = None
        self.backend_args: Optional[dict] = None


    def _load(self, results: dict) -> Optional[dict]:
        """Functions to load image.

        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        filename_rgb = results['img_path_rgb']
        filename_ir = results['img_path_ir']
        try:
            img_rgb = Image.open(filename_rgb).convert('RGB')
            img_ir = Image.open(filename_ir).convert('RGB')
        except Exception as e:
            if self.ignore_empty:
                return None
            else:
                raise e
        # in some cases, images are not read successfully, the img would be
        # `None`, refer to https://github.com/open-mmlab/mmpretrain/issues/1427
        assert img_rgb is not None, f'failed to load image: {filename_rgb}'
        assert img_ir is not None, f'failed to load image: {filename_ir}'
        
        results['img_rgb'] = img_rgb
        results['img_ir'] = img_ir
        results['img_shape'] = img_rgb.size[::-1] # (h, w)
        results['ori_shape'] = img_rgb.size[::-1]
        #results['scale_factor'] = 1.0
        return results

    def  __call__(self, results: dict) -> Optional[dict]:
        """Call function to load image and get image meta information.

        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.
        Returns:
            dict: The dict contains loaded image and meta information.
        """
        return self._load(results)
    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'ignore_empty={self.ignore_empty}, '
                    f'to_float32={self.to_float32}, '
                    f'backend_args={self.backend_args})')

        return repr_str
    

@TRANSFORMS.register()
class LoadAnnotations(object):
    """Load annotations for detection.
    Required Keys:
        - instances
    """

    def __init__(
            self,with_bbox=True, with_label=True, **kwargs) -> None:
        super(LoadAnnotations, self).__init__()
        self.with_bbox = with_bbox
        self.with_label = with_label
        

    def _load_bboxes(self, results: dict) -> None:
        """Private function to load bounding box annotations.

        Args:
            results (dict): Result dict from :obj:``mmengine.BaseDataset``.
        Returns:
            dict: The dict contains loaded bounding box annotations.
        """
        gt_bboxes = []
        gt_ignore_flags = []
        for instance in results.get('instances', []):
            gt_bboxes.append(instance['bbox'])
            gt_ignore_flags.append(instance['ignore_flag'])
        results['gt_bboxes'] = convert_to_tv_tensor(
                                    torch.tensor(gt_bboxes, dtype=torch.float32),
                                    key='boxes',
                                    box_format='xyxy',
                                    spatial_size=results['img_shape'])
        results['gt_ignore_flags'] = torch.from_numpy(np.array(gt_ignore_flags, dtype=bool))

    def _load_labels(self, results: dict) -> None:
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:``mmengine.BaseDataset``.

        Returns:
            dict: The dict contains loaded label annotations.
        """
        gt_bboxes_labels = []
        for instance in results.get('instances', []):
            gt_bboxes_labels.append(instance['bbox_label'])
        results['gt_bboxes_labels'] = torch.from_numpy(np.array(
            gt_bboxes_labels, dtype=np.int64))


    def __call__(self, results: dict) -> dict:
        """Function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:``mmengine.BaseDataset``.

        Returns:
            dict: The dict contains loaded bounding box, label and
            semantic segmentation.
        """
        if self.with_bbox:
            self._load_bboxes(results)
        if self.with_label:
            self._load_labels(results)

        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        return repr_str

@TRANSFORMS.register()
class RandomLoadText(object):

    def __init__(self,
                 text_path: str = None,
                 prompt_format: str = '{}',
                 num_neg_samples: Tuple[int, int] = (80, 80),
                 max_num_samples: int = 80,
                 padding_to_max: bool = False,
                 padding_value: str = '') -> None:
        self.prompt_format = prompt_format
        self.num_neg_samples = num_neg_samples
        self.max_num_samples = max_num_samples
        self.padding_to_max = padding_to_max
        self.padding_value = padding_value
        if text_path is not None:
            with open(text_path, 'r') as f:
                self.class_texts = json.load(f)

    def __call__(self, results: dict) -> dict:
        assert 'text' in results or hasattr(self, 'class_texts'), (
            'No texts found in results.')
        class_texts = results.get(
            'text',
            getattr(self, 'class_texts', None))
        num_classes = len(class_texts)
        if 'gt_labels' in results:
            gt_label_tag = 'gt_labels'
        elif 'gt_bboxes_labels' in results:
            gt_label_tag = 'gt_bboxes_labels'
        else:
            raise ValueError('No valid labels found in results.')
        positive_labels = set(results[gt_label_tag].tolist())
        if len(positive_labels) > self.max_num_samples:
            positive_labels = set(random.sample(list(positive_labels),
                                  k=self.max_num_samples))
        num_neg_samples = min(
            min(num_classes, self.max_num_samples) - len(positive_labels),
            random.randint(*self.num_neg_samples))
        candidate_neg_labels = []
        for idx in range(num_classes):
            if idx not in positive_labels:
                candidate_neg_labels.append(idx)
        negative_labels = random.sample(
            candidate_neg_labels, k=num_neg_samples)


        sampled_labels = list(positive_labels) + list(negative_labels)
        random.shuffle(sampled_labels)
        label2ids = {label: i for i, label in enumerate(sampled_labels)}

        gt_valid_mask = np.zeros(len(results['gt_bboxes']), dtype=bool)
        for idx, label in enumerate(results[gt_label_tag]):
            if label.item() in label2ids:
                gt_valid_mask[idx] = True
                results[gt_label_tag][idx] = label2ids[label.item()]
        results['gt_bboxes'] = results['gt_bboxes'][gt_valid_mask]
        results[gt_label_tag] = results[gt_label_tag][gt_valid_mask]
        if 'gt_masks' in results:
            results['gt_masks'] = results['gt_masks'][gt_valid_mask]
        if 'gt_ignore_flags' in results:
            results['gt_ignore_flags'] = results['gt_ignore_flags'][gt_valid_mask]
        if 'instances' in results:
            retaged_instances = []
            for idx, inst in enumerate(results['instances']):
                label = inst['bbox_label']
                if label in label2ids:
                    inst['bbox_label'] = label2ids[label]
                    retaged_instances.append(inst)
            results['instances'] = retaged_instances

        texts = []
        for label in sampled_labels:
            cls_caps = class_texts[label]
            # obj365 class_names can have multiple captions: e.g trash/bin 
            if '/' in cls_caps:
                cls_caps = cls_caps.split('/')
                # select one caption randomly for each class to avoid the performance drop caused by the long text input
                cls_caps = random.choice(cls_caps)
            texts.append(cls_caps)
        if self.padding_to_max:
            num_valid_labels = len(positive_labels) + len(negative_labels)
            num_padding = self.max_num_samples - num_valid_labels
            if num_padding > 0:
                texts += [self.padding_value] * num_padding
        results['text'] = tuple(texts)
        return results


@TRANSFORMS.register()
class MSPackDetInputs(object):
    """Pack the inputs data for the detection / semantic segmentation /
    panoptic segmentation.

    The ``img_meta`` item is always populated.  The contents of the
    ``img_meta`` dictionary depends on ``meta_keys``. By default this includes:

        - ``img_id``: id of the image

        - ``img_path``: path to the image file

        - ``ori_shape``: original shape of the image as a tuple (h, w)

        - ``img_shape``: shape of the image input to the network as a tuple \
            (h, w).  Note that images may be zero padded on the \
            bottom/right if the batch tensor is larger than this shape.

        - ``scale_factor``: a float indicating the preprocessing scale

        - ``flip``: a boolean indicating if image flip transform was used

        - ``flip_direction``: the flipping direction

    Args:
        meta_keys (Sequence[str], optional): Meta keys to be converted to
            ``mmcv.DataContainer`` and collected in ``data[img_metas]``.
            Default: ``('img_id', 'img_path', 'ori_shape', 'img_shape',
            'scale_factor', 'flip', 'flip_direction')``
    """
    mapping_table = {
        'gt_bboxes': 'bboxes',
        'gt_bboxes_labels': 'labels',
        'gt_masks': 'masks'
    }

    def __init__(self,
                 meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                            'scale_factor', 'flip', 'flip_direction')):
        self.meta_keys = meta_keys


    def __call__(self, results: dict) -> dict:
        """Method to pack the input data.

        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict:

            - 'inputs' (obj:`torch.Tensor`): The forward data of models.
            - 'data_sample' (obj:`DetDataSample`): The annotation info of the
                sample.
        """
        #print('results:', results)
        packed_results = dict()
        if 'img_rgb' in results and 'img_ir' in results:
            img_rgb = results['img_rgb']
            img_ir = results['img_ir']
            packed_results['inputs_rgb'] = img_rgb
            packed_results['inputs_ir'] = img_ir
        else:
            raise ValueError('The results must contain "img_rgb" and "img_ir" '
                             'keys for multispectral dataset.')

        if 'gt_ignore_flags' in results:
            valid_idx = np.where(results['gt_ignore_flags'] == 0)[0]
            ignore_idx = np.where(results['gt_ignore_flags'] == 1)[0]
        data_sample = {}
        instance_data = {}
        ignore_instance_data = {}

        
        for key in self.mapping_table.keys():
            if key not in results:
                continue
            if key == 'gt_bboxes':
                if 'gt_ignore_flags' in results:
                    instance_data[
                        self.mapping_table[key]] = results[key][valid_idx]
                    ignore_instance_data[
                        self.mapping_table[key]] = results[key][ignore_idx]
                else:
                    instance_data[self.mapping_table[key]] = results[key]
            else:
                if 'gt_ignore_flags' in results:
                    instance_data[self.mapping_table[key]] = to_tensor(
                        results[key][valid_idx])
                    ignore_instance_data[self.mapping_table[key]] = to_tensor(
                        results[key][ignore_idx])
                else:
                    instance_data[self.mapping_table[key]] = to_tensor(
                        results[key])
        data_sample['gt_instances'] = instance_data
        data_sample['ignored_instances'] = ignore_instance_data


        img_meta = {}
        for key in self.meta_keys:
            if key in results:
                img_meta[key] = results[key]
        data_sample['metainfo'] = img_meta
        packed_results['data_samples'] = data_sample

        return packed_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(meta_keys={self.meta_keys})'
        return repr_str


class Compose(object):
    def __init__(self, transforms: List[object]) -> None:
        self.transforms: List[Callable] = []
        if transforms is None:
            transforms = []
        
        for transform in transforms:
            if isinstance(transform, dict):
                transform = TRANSFORMS.build(transform)
            self.transforms.append(transform)

    def __call__(self, results: dict) -> dict:
        for transform in self.transforms:
            results = transform(results)
            if results is None:
                return None
        return results
    


