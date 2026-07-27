# Copyright (c) OpenMMLab. All rights reserved.
import copy, json, bisect
import os.path as osp
from typing import List, Union, Any
from torchvision.datasets.vision import VisionDataset
from .transforms.loading import Compose
import os
import random
import numpy as np
import torch

from pycocotools.coco import COCO
from util import DATASETS

__all__ = ['Few_shot_CocoDetection_RGBT', 'Support_RGBT', 'CocoDetection_RGBT', 'MSImageListDataset', 'ConcatDataset']

@DATASETS.register()
class Few_shot_CocoDetection_RGBT:
    """Dataset for COCO."""

    METAINFO = {
        'classes':
        ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
         'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
         'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
         'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
         'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
         'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
         'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
         'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
         'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
         'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
         'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
         'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
         'scissors', 'teddy bear', 'hair drier', 'toothbrush'),
        # palette is a list of color tuples, which is used for visualization.
        'palette':
        [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
         (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
         (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30),
         (165, 42, 42), (255, 77, 255), (0, 226, 252), (182, 182, 255),
         (0, 82, 0), (120, 166, 157), (110, 76, 0), (174, 57, 255),
         (199, 100, 0), (72, 0, 118), (255, 179, 240), (0, 125, 92),
         (209, 0, 151), (188, 208, 182), (0, 220, 176), (255, 99, 164),
         (92, 0, 73), (133, 129, 255), (78, 180, 255), (0, 228, 0),
         (174, 255, 243), (45, 89, 255), (134, 134, 103), (145, 148, 174),
         (255, 208, 186), (197, 226, 255), (171, 134, 1), (109, 63, 54),
         (207, 138, 255), (151, 0, 95), (9, 80, 61), (84, 105, 51),
         (74, 65, 105), (166, 196, 102), (208, 195, 210), (255, 109, 65),
         (0, 143, 149), (179, 0, 194), (209, 99, 106), (5, 121, 0),
         (227, 255, 205), (147, 186, 208), (153, 69, 1), (3, 95, 161),
         (163, 255, 0), (119, 0, 170), (0, 182, 199), (0, 165, 120),
         (183, 130, 88), (95, 32, 0), (130, 114, 135), (110, 129, 133),
         (166, 74, 118), (219, 142, 185), (79, 210, 114), (178, 90, 62),
         (65, 70, 15), (127, 167, 115), (59, 105, 106), (142, 108, 45),
         (196, 172, 0), (95, 54, 80), (128, 76, 255), (201, 57, 1),
         (246, 0, 122), (191, 162, 208)]
    }
    COCOAPI = COCO
    # ann_id is unique in coco dataset.
    ANN_ID_UNIQUE = True

    def __init__(self,
                 root: str,
                 ann_file: str,
                 data_prefix: str,
                 transforms=None,
                 support_transforms=None,
                 stage: str = 'meta_learning',
                 with_support: bool = True,
                 n_way: int = 5,
                 num_shots: int = 1,
                 min_area_support: int = 256,
                 key_rgb: str = None,
                 key_ir: str = None,
                 meta_info: dict = None,
                 return_classes: bool = True,
                 test_mode: bool = False,
                 filter_cfg: dict = None,
                 max_refetch: int = 100,
                 **kwargs) -> None:
        super().__init__()
        self.root = root
        self.transforms = Compose(transforms)
        self.key_rgb = key_rgb
        self.key_ir = key_ir
        self.ann_file = ann_file
        self.data_prefix = data_prefix.copy()
        self.data_prefix['img_rgb'] = osp.join(root, self.data_prefix['img_rgb'])
        self.data_prefix['img_ir'] = osp.join(root, self.data_prefix['img_ir'])
        self.return_classes = return_classes
        self.COCOAPI = COCO
        self.filter_cfg = filter_cfg
        self.test_mode = test_mode
        self.backend_args = kwargs.get('backend_args', None)
        self.metainfo = meta_info if meta_info is not None else self.METAINFO
        self.stage = stage
        self.n_way = n_way
        self.num_shots = num_shots
        self.support_pipeline = Compose(support_transforms)
        self.min_area_support = min_area_support
        self.with_support = with_support
        self.max_refetch = max_refetch

        self.data_list = self.load_data_list()

    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``

        Returns:
            List[dict]: A list of annotation.
        """  # noqa: E501
        self.coco = self.COCOAPI(osp.join(self.root, self.ann_file))
        # The order of returned `cat_ids` will not
        # change with the order of the `classes`
        if self.stage == 'meta_learning':
            self.cat_names_stage = self.metainfo['BASE_CLASSES']
        elif self.stage == 'few_shot_finetune':
            self.cat_names_stage = self.metainfo['ALL_CLASSES']
        else:
            raise ValueError(f'Invalid stage {self.stage} for few-shot RGBT'
                             ' detection dataset.')
        
        self.cat_names = self.metainfo['ALL_CLASSES'] # need to have the real mapping here
        self.cat_ids = self.coco.getCatIds(self.cat_names)
        self.cat_ids_stage = self.coco.getCatIds(self.cat_names_stage) # to filter annotations during meta-learning stage
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.cat2label_stage = {k: v for k, v in self.cat2label.items() if k in self.cat_ids_stage}
        self.cat_img_map = copy.deepcopy(self.coco.catToImgs)

        img_ids = self.coco.getImgIds()
        data_list = []
        total_ann_ids = []
        self.anns_per_cat = {i: [] for i, cat_id in enumerate(self.cat_ids)}

        for img_id in img_ids:
            raw_img_info = self.coco.loadImgs([img_id])[0]
            raw_img_info['img_id'] = img_id

            ann_ids = self.coco.getAnnIds(imgIds=[img_id])
            raw_ann_info = self.coco.loadAnns(ann_ids)
            total_ann_ids.extend(ann_ids)

            parsed_data_info = self.parse_data_info({
                'raw_ann_info':
                raw_ann_info,
                'raw_img_info':
                raw_img_info
            })
            if len(parsed_data_info['instances']) == 0:
                continue
            data_list.append(parsed_data_info)
        if self.ANN_ID_UNIQUE:
            assert len(set(total_ann_ids)) == len(
                total_ann_ids
            ), f"Annotation ids in '{self.ann_file}' are not unique!"

        del self.coco

        return data_list
    def _rand_another(self) -> int:
        """Get random index.

        Returns:
            int: Random index from 0 to ``len(self)-1``
        """
        return np.random.randint(0, len(self))
    
    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:
        """Parse raw annotation to target format.

        Args:
            raw_data_info (dict): Raw data information load from ``ann_file``

        Returns:
            Union[dict, List[dict]]: Parsed annotation.
        """
        img_info = raw_data_info['raw_img_info']
        ann_info = raw_data_info['raw_ann_info']

        data_info = {}

        # TODO: need to change data_prefix['img'] to data_prefix['img_path']
        img_path_rgb = osp.join(self.data_prefix['img_rgb'], img_info[self.key_rgb])
        img_path_ir = osp.join(self.data_prefix['img_ir'], img_info[self.key_ir])
        if self.data_prefix.get('seg', None):
            seg_map_path = osp.join(
                self.data_prefix['seg'],
                img_info['file_name'].rsplit('.', 1)[0] + self.seg_map_suffix)
        else:
            seg_map_path = None
        data_info['img_path_rgb'] = img_path_rgb
        data_info['img_path_ir'] = img_path_ir
        data_info['img_id'] = img_info['img_id']
        data_info['seg_map_path'] = seg_map_path
        data_info['height'] = img_info['height']
        data_info['width'] = img_info['width']


        instances = []
        for i, ann in enumerate(ann_info):
            instance = {}

            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids_stage:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]

            if ann.get('iscrowd', False):
                instance['ignore_flag'] = 1
            else:
                instance['ignore_flag'] = 0
            instance['bbox'] = bbox
            instance['bbox_label'] = self.cat2label[ann['category_id']]

            if ann.get('segmentation', None):
                instance['mask'] = ann['segmentation']

            instances.append(instance)
            if  self.with_support and ann['area'] >= self.min_area_support:
                info_cls = dict()
                info_cls['img_path_rgb'] = img_path_rgb
                info_cls['img_path_ir'] = img_path_ir
                info_cls['width'] = img_info['width']
                info_cls['height'] = img_info['height']
                info_cls['instances'] = [instance]
                self.anns_per_cat[self.cat2label[ann['category_id']]].append(info_cls)
        data_info['instances'] = instances
        return data_info

    def sample_support_set(self, query_cat_ids: List[int], num_shots: int) -> List[dict]:
        """Sample support set for a given category.
        Args:
            query_cat_id (int): Category id of the query instance.
            num_shots (int): Number of shots to sample.
        Returns:
            List[dict]: A list of support instances.
        """
        # sample n-way - len(query_cat_ids) classes to form negative samples
        cat_ids = list(self.cat2label_stage.values())
        if len(query_cat_ids) < self.n_way:
            neg_cat_ids = list(
                set(cat_ids) - set(query_cat_ids)
            )
            to_sample = min(self.n_way - len(query_cat_ids), len(neg_cat_ids))
            sampled_neg_cat_ids = random.sample(neg_cat_ids, to_sample)
            sampled_cat_ids = query_cat_ids + sampled_neg_cat_ids
        else:
            sampled_cat_ids = query_cat_ids

        random.shuffle(sampled_cat_ids)

        support_set = []
        for cat_id in sampled_cat_ids:
           infos = self.anns_per_cat[cat_id]
           sampled_infos = random.sample(infos, min(num_shots, len(infos)))
           support_set.extend(sampled_infos)


        return support_set

    def __getitem__(self, idx: int) -> dict:
        """Get the idx-th image and data information of dataset after
        ``self.pipeline``, and ``full_init`` will be called if the dataset has
        not been fully initialized.

        During training phase, if ``self.pipeline`` get ``None``,
        ``self._rand_another`` will be called until a valid image is fetched or
         the maximum limit of refetech is reached.

        Args:
            idx (int): The index of self.data_list.

        Returns:
            dict: The idx-th image and data information of dataset after
            ``self.pipeline``.
        """
        
        
        if self.test_mode:
            data = self.prepare_data(idx)
            if data is None:
                raise Exception('Test time pipline should not get `None` '
                                'data_sample')
            return data
        
        for _ in range(self.max_refetch + 1):
            data = self.prepare_data(idx)
            # Broken images or random augmentations may cause the returned data
            # to be None
            #print('data', data)
            if data is None:
                continue
            break
        if data is None:
            for _ in range(self.max_refetch + 1):
                idx = self._rand_another()
                data = self.prepare_data(idx)
                if data is not None:
                    break

        if self.with_support:
            query_cat_ids = data['data_samples']['gt_instances']['labels'].unique().tolist()
            support_set = self.sample_support_set(
                    query_cat_ids=query_cat_ids,
                    num_shots=self.num_shots
                    )
            Support_data_list = list()
            for i, support_info in enumerate(support_set):
                    # pipeline 
                    support_data = self.support_pipeline(copy.deepcopy(support_info))
                    Support_data_list.append(support_data)
            data['support_data'] = Support_data_list
        return data

    def prepare_data(self, idx) -> Any:
        """Get data processed by ``self.pipeline``.

        Args:
            idx (int): The index of ``data_info``.

        Returns:
            Any: Depends on ``self.pipeline``.
        """
        data_info = copy.deepcopy(self.data_list[idx])
        data = self.transforms(data_info)       

        return data
    
    def __len__(self) -> int:
        """Total number of samples of the dataset."""
        return len(self.data_list)
    




@DATASETS.register()
class Support_RGBT:
    """Query-free support dataset for RGBT few-shot detection, COCO format."""

    METAINFO = {
        'ALL_CLASSES':
        ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
         'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
         'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
         'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
         'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
         'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
         'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
         'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
         'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
         'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
         'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
         'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
         'scissors', 'teddy bear', 'hair drier', 'toothbrush'),
        # palette is a list of color tuples, which is used for visualization.
        'palette':
        [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
         (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
         (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30),
         (165, 42, 42), (255, 77, 255), (0, 226, 252), (182, 182, 255),
         (0, 82, 0), (120, 166, 157), (110, 76, 0), (174, 57, 255),
         (199, 100, 0), (72, 0, 118), (255, 179, 240), (0, 125, 92),
         (209, 0, 151), (188, 208, 182), (0, 220, 176), (255, 99, 164),
         (92, 0, 73), (133, 129, 255), (78, 180, 255), (0, 228, 0),
         (174, 255, 243), (45, 89, 255), (134, 134, 103), (145, 148, 174),
         (255, 208, 186), (197, 226, 255), (171, 134, 1), (109, 63, 54),
         (207, 138, 255), (151, 0, 95), (9, 80, 61), (84, 105, 51),
         (74, 65, 105), (166, 196, 102), (208, 195, 210), (255, 109, 65),
         (0, 143, 149), (179, 0, 194), (209, 99, 106), (5, 121, 0),
         (227, 255, 205), (147, 186, 208), (153, 69, 1), (3, 95, 161),
         (163, 255, 0), (119, 0, 170), (0, 182, 199), (0, 165, 120),
         (183, 130, 88), (95, 32, 0), (130, 114, 135), (110, 129, 133),
         (166, 74, 118), (219, 142, 185), (79, 210, 114), (178, 90, 62),
         (65, 70, 15), (127, 167, 115), (59, 105, 106), (142, 108, 45),
         (196, 172, 0), (95, 54, 80), (128, 76, 255), (201, 57, 1),
         (246, 0, 122), (191, 162, 208)],
         'BASE_CLASSES': ('truck', 'traffic light', 'fire hydrant', 'stop sign',
                  'parking meter', 'bench', 'elephant', 'bear', 'zebra',
                  'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                  'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                  'kite', 'baseball bat', 'baseball glove', 'skateboard',
                  'surfboard', 'tennis racket', 'wine glass', 'cup', 'fork',
                  'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
                  'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
                  'cake', 'bed', 'toilet', 'laptop', 'mouse', 'remote',
                  'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                  'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                  'teddy bear', 'hair drier', 'toothbrush'),
        'NOVEL_CLASSES': ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                   'train', 'boat', 'bird', 'cat', 'dog', 'horse', 'sheep',
                   'cow', 'bottle', 'chair', 'couch', 'potted plant',
                   'dining table', 'tv'),
    }

    COCOAPI = COCO
    # ann_id is unique in coco dataset.
    ANN_ID_UNIQUE = True
    _fully_initialized = True
    def __init__(self,
                 root: str,
                 ann_file: str,
                 data_prefix: str,
                 key_rgb: str,
                 key_ir: str,
                 transforms=None,
                 min_area_support: int = 256,
                 meta_info: dict = None,
                 return_classes: bool = True,
                 test_mode: bool = False,
                 filter_cfg: dict = None,
                 stage: str = 'meta_learning',
                 **kwargs) -> None:
        super().__init__()
        self.root = root
        self.transforms = Compose(transforms)
        self.key_rgb = key_rgb
        self.key_ir = key_ir
        self.ann_file = ann_file
        self.data_prefix = data_prefix.copy()
        self.data_prefix['img_rgb'] = osp.join(root, self.data_prefix['img_rgb'])
        self.data_prefix['img_ir'] = osp.join(root, self.data_prefix['img_ir'])
        self.return_classes = return_classes
        self.COCOAPI = COCO
        self.filter_cfg = filter_cfg
        self.test_mode = test_mode
        self.backend_args = kwargs.get('backend_args', None)
        self.metainfo = meta_info if meta_info is not None else self.METAINFO
        self.stage = stage
        self.min_area_support = min_area_support

        # Build COCO and cache anns per class
        self.coco = self.COCOAPI(osp.join(self.root, self.ann_file))

        if self.stage == 'meta_learning':
            self.cat_names_stage = self.metainfo['BASE_CLASSES']
        elif self.stage == 'few_shot_finetune':
            self.cat_names_stage = self.metainfo['ALL_CLASSES']
        else:
            raise ValueError(f'Invalid stage {self.stage} for few-shot RGBT'
                             ' detection dataset.')
        self.cat_names = self.metainfo['ALL_CLASSES'] # need to have the real mapping here
        self.cat_ids = self.coco.getCatIds(self.cat_names)
        self.cat_ids_stage = self.coco.getCatIds(self.cat_names_stage) # to filter annotations during meta-learning stage
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.cat_img_map = copy.deepcopy(self.coco.catToImgs)
        self.anns_per_cat = {i: [] for i, cat_id in enumerate(self.cat_ids)}

        img_ids = self.coco.getImgIds()
        support_img_ids = set()

        for img_id in img_ids:
            img_info = self.coco.loadImgs([img_id])[0]
            ann_ids = self.coco.getAnnIds(imgIds=[img_id])
            anns = self.coco.loadAnns(ann_ids)
            img_anns = []
            for ann in anns:
                if ann['category_id'] in self.cat_ids_stage and ann.get('area', 0) >= self.min_area_support:
                    self.anns_per_cat[self.cat2label[ann['category_id']]].append(
                        dict(img_info=img_info, ann=ann)
                    )
                    img_anns.append(ann)
            if len(img_anns) > 0:
                support_img_ids.add(img_id)
                del img_anns
        self.support_img_ids = list(support_img_ids)
        del self.coco


    def _sample_support_set(self):
        """Sample k-shot support per activated class."""
        support_infos = []
        for class_id in self.cat_ids_stage:
            anns = self.anns_per_cat[self.cat2label[class_id]]
            assert len(anns) >= 1, f"Not enough samples for class {class_id} to sample support set."
            s = random.sample(anns, 1)[0]
            instance = {}
            x1, y1, w, h = s['ann']['bbox']
            bbox = [x1, y1, x1 + w, y1 + h]
            instance['bbox'] = bbox
            instance['bbox_label'] = self.cat2label[s['ann']['category_id']]
            instance['mask'] = s['ann'].get('segmentation', None)
            if s['ann'].get('is_crowd', False):
                instance['ignore_flag'] = 1
            else:
                instance['ignore_flag'] = 0
            support_infos.append(
                    dict(
                        img_path_rgb=osp.join(self.data_prefix['img_rgb'], s['img_info'][self.key_rgb]),
                        img_path_ir=osp.join(self.data_prefix['img_ir'], s['img_info'][self.key_ir]),
                        height=s['img_info']['height'],
                        width=s['img_info']['width'],
                        instances=[instance]
                    )
                )
        return support_infos

    def __len__(self):
        return len(self.support_img_ids)
    
    def __getitem__(self, idx: int) -> dict:
        """Get the idx-th support data after ``self.support_pipeline``.

        Args:
            idx (int): The index of self.support_infos.

        Returns:
            dict: The idx-th support data after ``self.support_pipeline``.
        """
        support_info = self._sample_support_set()
        data = []
        for info in support_info:
            data.append(self.transforms(info)) 
        return data






@DATASETS.register()
class CocoDetection_RGBT:
    """Dataset for COCO."""

    METAINFO = {
        'classes':
        ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
         'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
         'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
         'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
         'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
         'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
         'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
         'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
         'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
         'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
         'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
         'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
         'scissors', 'teddy bear', 'hair drier', 'toothbrush'),
        # palette is a list of color tuples, which is used for visualization.
        'palette':
        [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
         (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
         (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30),
         (165, 42, 42), (255, 77, 255), (0, 226, 252), (182, 182, 255),
         (0, 82, 0), (120, 166, 157), (110, 76, 0), (174, 57, 255),
         (199, 100, 0), (72, 0, 118), (255, 179, 240), (0, 125, 92),
         (209, 0, 151), (188, 208, 182), (0, 220, 176), (255, 99, 164),
         (92, 0, 73), (133, 129, 255), (78, 180, 255), (0, 228, 0),
         (174, 255, 243), (45, 89, 255), (134, 134, 103), (145, 148, 174),
         (255, 208, 186), (197, 226, 255), (171, 134, 1), (109, 63, 54),
         (207, 138, 255), (151, 0, 95), (9, 80, 61), (84, 105, 51),
         (74, 65, 105), (166, 196, 102), (208, 195, 210), (255, 109, 65),
         (0, 143, 149), (179, 0, 194), (209, 99, 106), (5, 121, 0),
         (227, 255, 205), (147, 186, 208), (153, 69, 1), (3, 95, 161),
         (163, 255, 0), (119, 0, 170), (0, 182, 199), (0, 165, 120),
         (183, 130, 88), (95, 32, 0), (130, 114, 135), (110, 129, 133),
         (166, 74, 118), (219, 142, 185), (79, 210, 114), (178, 90, 62),
         (65, 70, 15), (127, 167, 115), (59, 105, 106), (142, 108, 45),
         (196, 172, 0), (95, 54, 80), (128, 76, 255), (201, 57, 1),
         (246, 0, 122), (191, 162, 208)]
    }
    COCOAPI = COCO
    # ann_id is unique in coco dataset.
    ANN_ID_UNIQUE = True

    def __init__(self,
                 root: str,
                 ann_file: str,
                 data_prefix: str,
                 transforms=None,
                 support_transforms=None,
                 stage: str = 'meta_learning',
                 with_support: bool = True,
                 n_way: int = 5,
                 num_shots: int = 1,
                 min_area_support: int = 256,
                 key_rgb: str = None,
                 key_ir: str = None,
                 meta_info: dict = None,
                 return_classes: bool = True,
                 test_mode: bool = False,
                 filter_cfg: dict = None,
                 max_refetch: int = 100,
                 **kwargs) -> None:
        super().__init__()
        self.root = root
        self.transforms = Compose(transforms)
        self.key_rgb = key_rgb
        self.key_ir = key_ir
        self.ann_file = ann_file
        self.data_prefix = data_prefix.copy()
        self.data_prefix['img_rgb'] = osp.join(root, self.data_prefix['img_rgb'])
        self.data_prefix['img_ir'] = osp.join(root, self.data_prefix['img_ir'])
        self.return_classes = return_classes
        self.COCOAPI = COCO
        self.filter_cfg = filter_cfg
        self.test_mode = test_mode
        self.backend_args = kwargs.get('backend_args', None)
        self.metainfo = meta_info if meta_info is not None else self.METAINFO
        self.stage = stage
        self.n_way = n_way
        self.num_shots = num_shots
        self.support_pipeline = Compose(support_transforms)
        self.min_area_support = min_area_support
        self.with_support = with_support
        self.max_refetch = max_refetch

        self.data_list = self.load_data_list()

    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``

        Returns:
            List[dict]: A list of annotation.
        """  # noqa: E501
        self.coco = self.COCOAPI(osp.join(self.root, self.ann_file))
        # The order of returned `cat_ids` will not
        # change with the order of the `classes`
        if self.stage == 'meta_learning':
            self.cat_names = self.metainfo['BASE_CLASSES']
        elif self.stage == 'few_shot_finetune':
            self.cat_names = self.metainfo['ALL_CLASSES']
        else:
            raise ValueError(f'Invalid stage {self.stage} for few-shot RGBT'
                             ' detection dataset.')
        
        self.cat_ids = self.coco.getCatIds(self.cat_names)
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.cat_img_map = copy.deepcopy(self.coco.catToImgs)

        img_ids = self.coco.getImgIds()
        data_list = []
        total_ann_ids = []
        self.anns_per_cat = {i: [] for i, cat_id in enumerate(self.cat_ids)}

        for img_id in img_ids:
            raw_img_info = self.coco.loadImgs([img_id])[0]
            raw_img_info['img_id'] = img_id

            ann_ids = self.coco.getAnnIds(imgIds=[img_id])
            raw_ann_info = self.coco.loadAnns(ann_ids)
            total_ann_ids.extend(ann_ids)

            parsed_data_info = self.parse_data_info({
                'raw_ann_info':
                raw_ann_info,
                'raw_img_info':
                raw_img_info
            })
            if len(parsed_data_info['instances']) == 0:
                continue
            data_list.append(parsed_data_info)
        if self.ANN_ID_UNIQUE:
            assert len(set(total_ann_ids)) == len(
                total_ann_ids
            ), f"Annotation ids in '{self.ann_file}' are not unique!"

        del self.coco

        return data_list

    def _rand_another(self) -> int:
        """Get random index.

        Returns:
            int: Random index from 0 to ``len(self)-1``
        """
        return np.random.randint(0, len(self))
    
    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:
        """Parse raw annotation to target format.

        Args:
            raw_data_info (dict): Raw data information load from ``ann_file``

        Returns:
            Union[dict, List[dict]]: Parsed annotation.
        """
        img_info = raw_data_info['raw_img_info']
        ann_info = raw_data_info['raw_ann_info']

        data_info = {}

        # TODO: need to change data_prefix['img'] to data_prefix['img_path']
        img_path_rgb = osp.join(self.data_prefix['img_rgb'], img_info[self.key_rgb])
        img_path_ir = osp.join(self.data_prefix['img_ir'], img_info[self.key_ir])
        if self.data_prefix.get('seg', None):
            seg_map_path = osp.join(
                self.data_prefix['seg'],
                img_info['file_name'].rsplit('.', 1)[0] + self.seg_map_suffix)
        else:
            seg_map_path = None
        data_info['img_path_rgb'] = img_path_rgb
        data_info['img_path_ir'] = img_path_ir
        data_info['img_id'] = img_info['img_id']
        data_info['seg_map_path'] = seg_map_path
        data_info['height'] = img_info['height']
        data_info['width'] = img_info['width']
        if self.return_classes:
            data_info['text'] = self.cat_names

        instances = []
        for i, ann in enumerate(ann_info):
            instance = {}

            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]

            if ann.get('iscrowd', False):
                instance['ignore_flag'] = 1
            else:
                instance['ignore_flag'] = 0
            instance['bbox'] = bbox
            instance['bbox_label'] = self.cat2label[ann['category_id']]

            if ann.get('segmentation', None):
                instance['mask'] = ann['segmentation']

            instances.append(instance)
            if  self.with_support and ann['area'] >= self.min_area_support:
                info_cls = dict()
                info_cls['img_path_rgb'] = img_path_rgb
                info_cls['img_path_ir'] = img_path_ir
                info_cls['width'] = img_info['width']
                info_cls['height'] = img_info['height']
                info_cls['instances'] = [instance]
                self.anns_per_cat[self.cat2label[ann['category_id']]].append(info_cls)
        data_info['instances'] = instances
        return data_info


    def __getitem__(self, idx: int) -> dict:
        """Get the idx-th image and data information of dataset after
        ``self.pipeline``, and ``full_init`` will be called if the dataset has
        not been fully initialized.

        During training phase, if ``self.pipeline`` get ``None``,
        ``self._rand_another`` will be called until a valid image is fetched or
         the maximum limit of refetech is reached.

        Args:
            idx (int): The index of self.data_list.

        Returns:
            dict: The idx-th image and data information of dataset after
            ``self.pipeline``.
        """
        
        
        if self.test_mode:
            data = self.prepare_data(idx)
            if data is None:
                raise Exception('Test time pipline should not get `None` '
                                'data_sample')
            return data
        
        for _ in range(self.max_refetch + 1):
            data = self.prepare_data(idx)
            # Broken images or random augmentations may cause the returned data
            # to be None
            #print('data', data)
            if data is None:
                continue
            break
        if data is None:
            for _ in range(self.max_refetch + 1):
                idx = self._rand_another()
                data = self.prepare_data(idx)
                if data is not None:
                    break
        return data

    def prepare_data(self, idx) -> Any:
        """Get data processed by ``self.pipeline``.

        Args:
            idx (int): The index of ``data_info``.

        Returns:
            Any: Depends on ``self.pipeline``.
        """
        data_info = copy.deepcopy(self.data_list[idx])
        data = self.transforms(data_info)       

        return data
    
    def __len__(self) -> int:
        """Total number of samples of the dataset."""
        return len(self.data_list)





@DATASETS.register()
class MSImageListDataset:
    """Dataset for loading a list of images.

    Args:
        img_list (list[str]): The list of image paths.
        data_prefix (dict, optional): The prefix of image path. Default to None.
        pipeline (list[dict], optional): The pipeline to process the images.
            Default to None.
    """
    
    def __init__(self,
                 data_root: str,
                 data_prefix = None,
                 pipeline = None) -> None:
        self.data_prefix = data_prefix
        self.pipeline = Compose(pipeline)
        self.data = self.load_data_list(data_root)

    def load_data_list(self, data_root: str):
        img_folder_rgb = osp.join(data_root, self.data_prefix['img_rgb'])
        img_folder_ir  = osp.join(data_root, self.data_prefix['img_ir'])

        common = sorted(set(os.listdir(img_folder_rgb)) & set(os.listdir(img_folder_ir)))

        img_pairs = [
            (osp.join(img_folder_rgb, name),
            osp.join(img_folder_ir, name))
            for name in common
        ]
        return img_pairs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        results = dict(img_path_rgb=img_path[0], img_path_ir=img_path[1])
        results = self.pipeline(copy.deepcopy(results))
        return results


@DATASETS.register()
class ConcatDataset:
    """A wrapper of concatenating multiple datasets.

    Args:
        datasets (list[:obj:`BaseDataset`]): The datasets to be concatenated.
    """

    def __init__(self, datasets):
        self.datasets = []
        for dataset in datasets:
            dataset = DATASETS.build(dataset)
            self.datasets.append(dataset)
        self.cumulative_sizes = self.cumsum(self.datasets)

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]