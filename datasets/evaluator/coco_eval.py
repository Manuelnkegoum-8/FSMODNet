# Copyright (c) OpenMMLab. All rights reserved.
import datetime, warnings, json
import itertools
import os.path as osp
import tempfile
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Union
import numpy as np
import torch
from terminaltables import AsciiTable
from pycocotools.coco import COCO
from util import METRICS
from util.misc import is_main_process
from lightning.pytorch.utilities.rank_zero import rank_zero_info
from pycocotools.cocoeval import COCOeval
import torch.distributed as dist
import logging

logger = logging.getLogger(__name__)

__all__ = ['CocoMetric']


@METRICS.register()
class CocoMetric:
    """COCO evaluation metric.

    Evaluate AR, AP, and mAP for detection tasks including proposal/box
    detection and instance segmentation. Please refer to
    https://cocodataset.org/#detection-eval for more details.

    Args:
        ann_file (str, optional): Path to the coco format annotation file.
            If not specified, ground truth annotations from the dataset will
            be converted to coco format. Defaults to None.
        metric (str | List[str]): Metrics to be evaluated. Valid metrics
            include 'bbox', 'segm', 'proposal', and 'proposal_fast'.
            Defaults to 'bbox'.
        classwise (bool): Whether to evaluate the metric class-wise.
            Defaults to False.
        proposal_nums (Sequence[int]): Numbers of proposals to be evaluated.
            Defaults to (100, 300, 1000).
        iou_thrs (float | List[float], optional): IoU threshold to compute AP
            and AR. If not specified, IoUs from 0.5 to 0.95 will be used.
            Defaults to None.
        metric_items (List[str], optional): Metric result names to be
            recorded in the evaluation result. Defaults to None.
        format_only (bool): Format the output results without perform
            evaluation. It is useful when you want to format the result
            to a specific format and submit it to the test server.
            Defaults to False.
        outfile_prefix (str, optional): The prefix of json files. It includes
            the file path and the prefix of filename, e.g., "a/b/prefix".
            If not specified, a temp file will be created. Defaults to None.
        file_client_args (dict, optional): Arguments to instantiate the
            corresponding backend in mmdet <= 3.0.0rc6. Defaults to None.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
        sort_categories (bool): Whether sort categories in annotations. Only
            used for `Objects365V1Dataset`. Defaults to False.
        use_mp_eval (bool): Whether to use mul-processing evaluation
    """
    default_prefix: Optional[str] = 'coco'

    def __init__(self,
                 ann_file: Optional[str] = None,
                 stage='meta_learning',
                 dataset_meta: Optional[dict] = None,
                 metric: Union[str, List[str]] = 'bbox',
                 classwise: bool = False,
                 proposal_nums: Sequence[int] = (100, 300, 1000),
                 iou_thrs: Optional[Union[float, Sequence[float]]] = None,
                 metric_items: Optional[Sequence[str]] = None,
                 format_only: bool = False,
                 outfile_prefix: Optional[str] = './',
                 file_client_args: dict = None,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 sort_categories: bool = False,
                 use_mp_eval: bool = False) -> None:

        # coco evaluation metrics
        self.metrics = metric if isinstance(metric, list) else [metric]
        self.stage = stage
        allowed_metrics = ['bbox']
        for metric in self.metrics:
            if metric not in allowed_metrics:
                raise KeyError(
                    "metric should be one of 'bbox'"
                    f"'proposal_fast', but got {metric}.")

        # do class wise evaluation, default False
        self.classwise = classwise
        # whether to use multi processing evaluation, default False
        self.use_mp_eval = use_mp_eval

        # proposal_nums used to compute recall or precision.
        self.proposal_nums = list(proposal_nums)

        # iou_thrs used to compute recall or precision.
        if iou_thrs is None:
            iou_thrs = np.linspace(
                .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        self.iou_thrs = iou_thrs
        self.metric_items = metric_items
        self.format_only = format_only
        if self.format_only:
            assert outfile_prefix is not None, 'outfile_prefix must be not'
            'None when format_only is True, otherwise the result files will'
            'be saved to a temp directory which will be cleaned up at the end.'

        self.outfile_prefix = outfile_prefix

        self.dataset_meta = dataset_meta
        if ann_file is None:
            self._coco_api = None
            if is_main_process():
                warnings.warn("ann_file is None. The ground truth data instances will be used to initialize the cocoApi \
                to perform evaluation.")
        else:
            self._coco_api = COCO(ann_file)
        self.classwise = classwise
        self.results = []
        #self.cat_ids = self.coco_gt.getCatIds() if self.coco_gt is not None else None
        #self.img_ids = self.coco_gt.getImgIds() if self.coco_gt is not None else None

        # handle dataset lazy init
        self.cat_ids = None
        self.img_ids = None

    def xyxy2xywh(self, bbox: np.ndarray) -> list:
        """Convert ``xyxy`` style bounding boxes to ``xywh`` style for COCO
        evaluation.

        Args:
            bbox (numpy.ndarray): The bounding boxes, shape (4, ), in
                ``xyxy`` order.

        Returns:
            list[float]: The converted bounding boxes, in ``xywh`` order.
        """

        _bbox: List = bbox.tolist()
        return [
            _bbox[0],
            _bbox[1],
            _bbox[2] - _bbox[0],
            _bbox[3] - _bbox[1],
        ]

    def results2json(self, results: Sequence[dict],
                     outfile_prefix: str) -> dict:
        """Dump the detection results to a COCO style json file.

        There are 3 types of results: proposals, bbox predictions, mask
        predictions, and they have different data types. This method will
        automatically recognize the type, and dump them to json files.

        Args:
            results (Sequence[dict]): Testing results of the
                dataset.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json files will be named
                "somepath/xxx.bbox.json", "somepath/xxx.segm.json",
                "somepath/xxx.proposal.json".

        Returns:
            dict: Possible keys are "bbox", "segm", "proposal", and
            values are corresponding filenames.
        """
        bbox_json_results = []
        segm_json_results = [] if 'masks' in results[0] else None
        for idx, result in enumerate(results):
            image_id = result.get('img_id', idx)
            labels = result['labels']
            bboxes = result['bboxes']
            scores = result['scores']
            text = result.get('text', None)

            # bbox results
            for i, label in enumerate(labels):
                data = dict()
                data['image_id'] = image_id
                data['bbox'] = self.xyxy2xywh(bboxes[i])
                data['score'] = float(scores[i])
                if text is not None:
                    data['category_id'] = self._coco_api.getCatIds([text[label]])[0]
                else:
                    data['category_id'] = self.cat_ids[label]
                bbox_json_results.append(data)

        result_files = dict()
        result_files['bbox'] = f'{outfile_prefix}bbox.json'
        json.dump(bbox_json_results, open(result_files['bbox'], 'w'))

        return result_files


    # TODO: data_batch is no longer needed, consider adjusting the
    #  parameter position
    def process(self,data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of data samples that
                contain annotations and predictions.
        """
        for data_sample in data_samples:
            result = dict()
            pred = data_sample['pred_instances']
            metainfo = data_sample['metainfo']
            result['img_id'] = metainfo['img_id']
            result['bboxes'] = pred['bboxes'].cpu().numpy()
            result['scores'] = pred['scores'].cpu().numpy()
            result['labels'] = pred['labels'].cpu().numpy()
            

            # parse gt
            gt = dict()
            gt['width'] = metainfo['ori_shape'][1]
            gt['height'] = metainfo['ori_shape'][0]
            gt['img_id'] = metainfo['img_id']
            result['text'] = metainfo.get('text', None)
            if self._coco_api is None:
                # TODO: Need to refactor to support LoadAnnotations
                assert 'gt_instances' in data_sample, \
                    'ground truth is required for evaluation when ' \
                    '`ann_file` is not provided'
                gt['anns'] = data_sample['gt_instances']
            # add converted result to the results list
            self.results.append((gt, result))
    
    def synchronize_between_processes(self) -> None:
        """Synchronize results between different processes when distributed
        evaluation is performed. The results from different processes should be
        gathered and merged into ``self.results``."""

        if not dist.is_available() or not dist.is_initialized():
            return 
        
        world_size = dist.get_world_size()
        if world_size == 1:
            return
        # gather results from all ranks
        results_list = [None] * dist.get_world_size()
        dist.all_gather_object(results_list, self.results)
        # merge results
        merged_results = []
        for res in results_list:
            merged_results.extend(res)
        self.results = merged_results


    def compute_metrics(self) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        #logger: MMLogger = MMLogger.get_current_instance()
        results = self.results
        # split gt and prediction list
        gts, preds = zip(*results)

        tmp_dir = None
        if self.outfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            outfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            outfile_prefix = self.outfile_prefix

        if self._coco_api is None:
            # use converted gt json file to initialize coco api
            #logger.info('Converting ground truth to coco format...')
            coco_json_path = self.gt_to_coco_json(
                gt_dicts=gts, outfile_prefix=outfile_prefix)
            self._coco_api = COCO(coco_json_path)

        cat_names = self.dataset_meta['ALL_CLASSES']
        
        self.cat_ids = self._coco_api.getCatIds(cat_names)
        # names 
        
        if self.img_ids is None:
            self.img_ids = self._coco_api.getImgIds()

        # convert predictions to coco format and dump to json file
        result_files = self.results2json(preds, outfile_prefix)

        eval_results = OrderedDict()
        if self.format_only:
            #logger.info('results are saved in '
            #            f'{osp.dirname(outfile_prefix)}')
            return eval_results

        for metric in self.metrics:
            # evaluate proposal, bbox and segm
            iou_type = 'bbox' if metric == 'proposal' else metric
            if metric not in result_files:
                raise KeyError(f'{metric} is not in results')
            try:
                predictions = result_files[metric]
                coco_dt = self._coco_api.loadRes(predictions)

            except IndexError:
                
                break

            coco_eval = COCOeval(self._coco_api, coco_dt, iou_type)
            coco_eval.params.catIds = self.cat_ids
            coco_eval.params.imgIds = self.img_ids
            coco_eval.params.maxDets = list(self.proposal_nums)
            coco_eval.params.iouThrs = self.iou_thrs

            # mapping of cocoEval.stats
            coco_metric_names = {
                'mAP': 0,
                'mAP_50': 1,
                'mAP_75': 2,
                'mAP_s': 3,
                'mAP_m': 4,
                'mAP_l': 5,
                'AR@100': 6,
                'AR@300': 7,
                'AR@1000': 8,
                'AR_s@1000': 9,
                'AR_m@1000': 10,
                'AR_l@1000': 11
            }
            metric_items = self.metric_items
            if metric_items is not None:
                for metric_item in metric_items:
                    if metric_item not in coco_metric_names:
                        raise KeyError(
                            f'metric item "{metric_item}" is not supported')

            if metric == 'proposal':
                coco_eval.params.useCats = 0
                coco_eval.evaluate()
                coco_eval.accumulate()
                coco_eval.summarize()
                if metric_items is None:
                    metric_items = [
                        'AR@100', 'AR@300', 'AR@1000', 'AR_s@1000',
                        'AR_m@1000', 'AR_l@1000'
                    ]

                for item in metric_items:
                    val = float(
                        f'{coco_eval.stats[coco_metric_names[item]]:.3f}')
                    eval_results[item] = val
            else:
                coco_eval.evaluate()
                coco_eval.accumulate()
                coco_eval.summarize()
                class_wise_map50 = {}
                base_map_50 = dict(category='base_map_50',map=[], map_50=[], map_75=[], map_s=[], map_m=[], map_l=[])
                novel_map_50 = dict(category='novel_map_50',map=[], map_50=[], map_75=[], map_s=[], map_m=[], map_l=[])
                if self.classwise:  # Compute per-category AP
                    # Compute per-category AP
                    # from https://github.com/facebookresearch/detectron2/
                    precisions = coco_eval.eval['precision']
                    # precision: (iou, recall, cls, area range, max dets)
                    assert len(self.cat_ids) == precisions.shape[2]

                    results_per_category = []
                    for idx, cat_id in enumerate(self.cat_ids):
                        t = []
                        # area range index 0: all area ranges
                        # max dets index -1: typically 100 per image
                        nm = self._coco_api.loadCats(cat_id)[0]
                        precision = precisions[:, :, idx, 0, -1]
                        precision = precision[precision > -1]
                        if precision.size:
                            ap = np.mean(precision)
                        else:
                            ap = float('nan')
                        t.append(f'{nm["name"]}')
                        t.append(f'{round(ap, 3)}')
                        eval_results[f'{nm["name"]}_precision'] = round(ap, 3)
                        if nm["name"] in self.dataset_meta['BASE_CLASSES']:
                            base_map_50['map'].append(round(ap, 3))
                        elif nm["name"] in self.dataset_meta['NOVEL_CLASSES']:
                            novel_map_50['map'].append(round(ap, 3))
                        else:
                            pass

                        # indexes of IoU  @50 and @75
                        for count, iou in enumerate([0, 5]):
                            precision = precisions[iou, :, idx, 0, -1]
                            precision = precision[precision > -1]
                            if precision.size:
                                ap = np.mean(precision)
                            else:
                                ap = float('nan')
                            t.append(f'{round(ap, 3)}')

                            if count == 0:
                                class_wise_map50[nm["name"]] = round(ap, 3)
                            
                            if nm["name"] in self.dataset_meta['BASE_CLASSES']:
                                if count == 0:
                                    base_map_50['map_50'].append(round(ap, 3))
                                elif count == 1:
                                    base_map_50['map_75'].append(round(ap, 3))
                            elif nm["name"] in self.dataset_meta['NOVEL_CLASSES']:
                                if count == 0:
                                    novel_map_50['map_50'].append(round(ap, 3))
                                elif count == 1:
                                    novel_map_50['map_75'].append(round(ap, 3))
                            else:
                                pass

                        # indexes of area of small, median and large
                        for area in [1, 2, 3]:
                            precision = precisions[:, :, idx, area, -1]
                            precision = precision[precision > -1]
                            if precision.size:
                                ap = np.mean(precision)
                            else:
                                ap = float('nan')
                            t.append(f'{round(ap, 3)}')
                            if nm["name"] in self.dataset_meta['BASE_CLASSES']:
                                if area == 1:
                                    base_map_50['map_s'].append(round(ap, 3))
                                elif area == 2:
                                    base_map_50['map_m'].append(round(ap, 3))
                                elif area == 3:
                                    base_map_50['map_l'].append(round(ap, 3))
                            elif nm["name"] in self.dataset_meta['NOVEL_CLASSES']:
                                if area == 1:
                                    novel_map_50['map_s'].append(round(ap, 3))
                                elif area == 2:
                                    novel_map_50['map_m'].append(round(ap, 3))
                                elif area == 3:
                                    novel_map_50['map_l'].append(round(ap, 3))
                            else:
                                pass
                        results_per_category.append(tuple(t))
                        
                    # mean over base classes and novel classes
                    base_map_50 = {k: round(np.mean(v), 3) if isinstance(v, list) else v for k, v in base_map_50.items()}
                    results_per_category.append(tuple(base_map_50.values()))

                    if len(novel_map_50['map_50']) > 0:
                        novel_map_50 = {k: round(np.mean(v), 3) if isinstance(v, list) else v for k, v in novel_map_50.items()}
                        results_per_category.append(tuple(novel_map_50.values()))
                    num_columns = len(results_per_category[0])
                    results_flatten = list(
                        itertools.chain(*results_per_category))
                    headers = [
                        'category', 'mAP', 'mAP_50', 'mAP_75', 'mAP_s',
                        'mAP_m', 'mAP_l'
                    ]
                    results_2d = itertools.zip_longest(*[
                        results_flatten[i::num_columns]
                        for i in range(num_columns)
                    ])
                    table_data = [headers]
                    table_data += [result for result in results_2d]
                    table = AsciiTable(table_data)
                    #logger.info('\n' + table.table)
                    logger.info('\n' + table.table)

                if metric_items is None:
                    metric_items = [
                        'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l'
                    ]

                for metric_item in metric_items:
                    key = f'{metric}_{metric_item}'
                    val = coco_eval.stats[coco_metric_names[metric_item]]
                    eval_results[key] = float(f'{round(val, 3)}')

                ap = coco_eval.stats[:6]
                logger.info(f'{metric}_mAP_copypaste: {ap[0]:.3f} '
                            f'{ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
                            f'{ap[4]:.3f} {ap[5]:.3f}')

        if tmp_dir is not None:
            tmp_dir.cleanup()
        if self.stage == 'few_shot_finetune':
            eval_results.update({'base_map_50': base_map_50['map_50'], 'novel_map_50': novel_map_50['map_50']})
        else:
            eval_results.update({'base_map_50': base_map_50['map_50']})
        
        # add the default prefix to metric names to avoid name conflicts of different evaluators
        if self.default_prefix is not None:
            eval_results = {f'{self.default_prefix}/{k}': v for k, v in eval_results.items()}

        return eval_results