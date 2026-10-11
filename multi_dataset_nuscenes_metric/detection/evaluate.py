# nuScenes dev-kit.
# Code written by Holger Caesar & Oscar Beijbom, 2018.
# Modified for the MAGneT-3D cross-dataset benchmark.

import argparse
import json
import os
import random
import time
from typing import Tuple, Dict, Any
from collections import defaultdict
import copy
import numpy as np
import sys

import pycocotools.coco as coco

from nuscenes.eval.detection.algo import calc_ap, calc_tp
from multi_dataset_nuscenes_metric.detection.constants import TP_METRICS
from nuscenes.eval.detection.data_classes import DetectionMetrics, DetectionMetricDataList

from multi_dataset_nuscenes_metric.common.config import config_factory
from multi_dataset_nuscenes_metric.common.loaders import load_boxes
from multi_dataset_nuscenes_metric.detection.data_classes import DetectionConfig, DetectionBox, WeightedDetectionMetrics
from multi_dataset_nuscenes_metric.detection.render import summary_plot, class_pr_curve, class_tp_curve, dist_pr_curve, \
    visualize_sample
from multi_dataset_nuscenes_metric.detection.algo import accumulate


class DetectionEval:
    """
    This is a modified version of the nuScenes detection evaluation code.
    Results are written to the provided output_dir.
    nuScenes uses the following detection metrics:
    - Mean Average Precision (mAP): Uses center-distance as matching criterion; averaged over distance thresholds.
    - True Positive (TP) metrics: Average of translation, scale and orientation errors.
    - nuScenes Detection Score (NDS): The weighted sum of the above.
    Here is an overview of the functions in this method:
    - init: Loads GT annotations and predictions.
    - run: Performs evaluation and dumps the metric data to disk.
    - render: Renders various plots and dumps to disk.
    We assume that:
    - Every sample_token is given in the results, although there may be not predictions for that sample.
    Please see https://www.nuscenes.org/object-detection for more details.
    """
    def __init__(self, annotation: coco.COCO, result: coco.COCO, config: DetectionConfig, output_dir: str = None,
                 verbose: bool = True, class_weighted_metric: bool = True, with_velocity: bool = False,
                 tp_error_averaging: str = 'recall'):
        """
        Initialize a DetectionEval object.
        :param annotation: The annotation file in COCO format.
        :param result: The result file in COCO format.
        :param config: A DetectionConfig object.
        :param output_dir: Folder to save plots and results to.
        :param verbose: Whether to print to stdout.
        :param class_weighted_metric: Weight the classes by their number of gt boxes.
        :param with_velocity: Include the velocity error in NDS.
        :param tp_error_averaging: 'recall' or 'all_tp', see :func:`accumulate`.
        """
        self.annotation = annotation
        self.result = result
        self.output_dir = output_dir
        self.verbose = verbose
        self.cfg = config

        self.plot_dir = os.path.join(self.output_dir, 'plots')
        os.makedirs(self.plot_dir, exist_ok=True)

        # Make dirs.
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)

        # Load data.
        self.log('Initializing nuScenes detection evaluation')
        self.pred_boxes = load_boxes(self.result, self.annotation.dataset['categories'],
                                     self.annotation.dataset['attributes'], self.cfg.max_boxes_per_sample, DetectionBox,
                                     gt=False, verbose=verbose)
        self.gt_boxes = load_boxes(self.annotation, self.annotation.dataset['categories'],
                                   self.annotation.dataset['attributes'], self.cfg.max_boxes_per_sample, DetectionBox,
                                   gt=True, verbose=verbose)

        assert set(self.pred_boxes.sample_tokens) == set(self.gt_boxes.sample_tokens), \
            "Samples in split doesn't match samples in predictions."

        self.sample_tokens = self.gt_boxes.sample_tokens
        self.class_weighted_metric = class_weighted_metric
        self.with_velocity = with_velocity
        self.tp_error_averaging = tp_error_averaging

    def log(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def evaluate(self) -> Tuple[DetectionMetrics, DetectionMetricDataList]:
        """
        Performs the actual evaluation.
        :return: A tuple of high-level and the raw metric data.
        """
        start_time = time.time()

        # -----------------------------------
        # Step 1: Accumulate metric data for all classes and distance thresholds.
        # -----------------------------------
        self.log('Accumulating metric data...')
        metric_data_list = DetectionMetricDataList()
        for class_name in self.cfg.class_names:
            for dist_th in self.cfg.dist_ths:
                md = accumulate(self.gt_boxes, self.pred_boxes, class_name, self.cfg.dist_fcn_callable, dist_th,
                                tp_error_averaging=self.tp_error_averaging)
                metric_data_list.set(class_name, dist_th, md)

        # -----------------------------------
        # Step 2: Calculate metrics from the data.
        # -----------------------------------
        self.log('Calculating metrics...')
        metrics = WeightedDetectionMetrics.from_boxes(self.cfg, self.gt_boxes, class_weighted_metric=self.class_weighted_metric,
                                                      with_velocity=self.with_velocity)
        for class_name in self.cfg.class_names:
            # Compute APs.
            for dist_th in self.cfg.dist_ths:
                metric_data = metric_data_list[(class_name, dist_th)]
                ap = calc_ap(metric_data, self.cfg.min_recall, self.cfg.min_precision)
                metrics.add_label_ap(class_name, dist_th, ap)

            # Compute TP metrics.
            for metric_name in TP_METRICS:
                metric_data = metric_data_list[(class_name, self.cfg.dist_th_tp)]
                tp = calc_tp(metric_data, self.cfg.min_recall, metric_name)
                metrics.add_label_tp(class_name, metric_name, tp)

        # Compute evaluation time.
        metrics.add_runtime(time.time() - start_time)

        return metrics, metric_data_list

    def render(self, metrics: DetectionMetrics, md_list: DetectionMetricDataList) -> None:
        """
        Renders various PR and TP curves.
        :param metrics: DetectionMetrics instance.
        :param md_list: DetectionMetricDataList instance.
        """
        self.log('Rendering PR and TP curves')

        def savepath(name):
            return os.path.join(self.plot_dir, name + '.pdf')

        summary_plot(md_list, metrics, min_precision=self.cfg.min_precision, min_recall=self.cfg.min_recall,
                     dist_th_tp=self.cfg.dist_th_tp, savepath=savepath('summary'))

        for detection_name in self.cfg.class_names:
            class_pr_curve(md_list, metrics, detection_name, self.cfg.min_precision, self.cfg.min_recall,
                           savepath=savepath(detection_name + '_pr'))

            class_tp_curve(md_list, metrics, detection_name, self.cfg.min_recall, self.cfg.dist_th_tp,
                           savepath=savepath(detection_name + '_tp'))

        for dist_th in self.cfg.dist_ths:
            dist_pr_curve(md_list, metrics, dist_th, self.cfg.min_precision, self.cfg.min_recall,
                          savepath=savepath('dist_pr_' + str(dist_th)))

    def main(self, plot_examples: int = 0, render_curves: bool = True) -> Dict[str, Any]:
        """
        Main function that loads the evaluation code, visualizes samples, runs the evaluation and renders stat plots.
        :param plot_examples: How many example visualizations to write to disk.
        :param render_curves: Whether to render PR and TP curves to disk.
        :return: A dict that stores the high-level metrics and meta data.
        """
        if plot_examples > 0:
            if not os.path.isdir(self.plot_dir):
                os.makedirs(self.plot_dir)

            # Select a random but fixed subset to plot.
            random.seed(42)
            sample_tokens = list(self.sample_tokens)
            random.shuffle(sample_tokens)
            sample_tokens = sample_tokens[:plot_examples]

            # Visualize samples.
            example_dir = os.path.join(self.output_dir, 'examples')
            if not os.path.isdir(example_dir):
                os.mkdir(example_dir)
            for sample_token in sample_tokens:
                visualize_sample(sample_token, self.gt_boxes, self.pred_boxes, verbose=self.verbose,
                                 savepath=os.path.join(example_dir, '{}.png'.format(
                                     sample_token.strip(os.sep).replace(os.sep, '_'))))

        # Run evaluation.
        metrics, metric_data_list = self.evaluate()

        # Render PR and TP curves.
        if render_curves:
            self.render(metrics, metric_data_list)

        # Dump the metric data, meta and metrics to disk.
        self.log('Saving metrics to: %s' % self.output_dir)
        metrics_summary = metrics.serialize()
        with open(os.path.join(self.output_dir, 'metrics_summary.json'), 'w') as f:
            json.dump(metrics_summary, f, indent=2)
        with open(os.path.join(self.output_dir, 'metrics_details.json'), 'w') as f:
            json.dump(metric_data_list.serialize(), f, indent=2)

        # Print high-level metrics.
        self.log('mAP: %.4f' % (metrics_summary['mean_ap']))
        err_name_mapping = {'trans_err': 'mATE', 'scale_err': 'mASE', 'orient_err': 'mAOE' ,'vel_err': 'mAVE'}
        for tp_name, tp_val in metrics_summary['tp_errors'].items():
            self.log('%s: %.4f' % (err_name_mapping[tp_name], tp_val))
        self.log('NDS: %.4f' % (metrics_summary['nd_score']))
        self.log('Eval time: %.1fs' % metrics_summary['eval_time'])

        # Print per-class metrics.
        self.log()
        self.log('Per-class results:')
        self.log('{:<14s}{:<10s}{:<10s}{:<10s}{:<10s}{:<10s}'.format('Object Class', 'AP', 'ATE', 'ASE', 'AOE', 'AAE'))
        class_aps = metrics_summary['mean_dist_aps']
        class_tps = metrics_summary['label_tp_errors']
        for class_name in class_aps.keys():
            self.log('{:<14s}{:<10.3f}{:<10.3f}{:<10.3f}{:<10.3f}{:<10.3f}'.format(class_name, class_aps[class_name],
                                                                          class_tps[class_name]['trans_err'],
                                                                          class_tps[class_name]['scale_err'],
                                                                          class_tps[class_name]['orient_err'],
                                                                          class_tps[class_name]['vel_err']))

        return metrics_summary


def evaluate(annotation_or_annotation_path, result_or_result_path, save_dir: str, plot_examples: int = 10, render_curves: int = 1,
             verbose: int = 1, class_weighted_metric: bool = True, with_velocity: bool = False,
             tp_error_averaging: str = 'recall') -> Dict[str, Any]:
    # Settings.
    output_dir_ = os.path.join(os.path.expanduser(save_dir), 'dsc_eval_det_output')
    plot_examples_ = plot_examples
    render_curves_ = bool(render_curves)
    verbose_ = bool(verbose)

    if isinstance(annotation_or_annotation_path, str):
        annotation_or_annotation_path = os.path.expanduser(annotation_or_annotation_path)
        ann_ = coco.COCO(annotation_or_annotation_path)
    else:
        ann_ = FileLessCOCO(annotation_or_annotation_path)

    if isinstance(result_or_result_path, str):
        result_or_result_path = os.path.expanduser(result_or_result_path)
    if isinstance(result_or_result_path, list) and len(result_or_result_path) == 0:
        # no predictions: every image has an empty result
        res_ = FileLessCOCO(dict(images=ann_.dataset['images'], annotations=[],
                                 categories=ann_.dataset['categories']))
    else:
        res_ = ann_.loadRes(result_or_result_path)

    cfg_ = config_factory(ann_, DetectionConfig)

    eval = DetectionEval(annotation=ann_, result=res_, config=cfg_, output_dir=output_dir_, verbose=verbose_,
                         class_weighted_metric=class_weighted_metric, with_velocity=with_velocity,
                         tp_error_averaging=tp_error_averaging)
    metrics_summary = eval.main(plot_examples=plot_examples_, render_curves=render_curves_)
    return json.loads(json.dumps(metrics_summary))


class FileLessCOCO(coco.COCO): 
    def __init__(self, dataset=None):
        # load dataset
        self.dataset,self.anns,self.cats,self.imgs = dataset, dict(), dict(), dict()
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)
        self.createIndex()


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def multi_dataset_evaluation(gt_annotation_path: str, preds: list, save_dir: str, class_weighted_metric: bool = False,
                             only_predicted_imgs: bool = False, with_velocity: bool = False,
                             tp_error_averaging: str = 'recall') -> Dict[str, Any]:
    os.makedirs(save_dir, exist_ok=True)
    with open(gt_annotation_path, 'r') as file: 
        gt = json.load(file)
   
    # GTs       
    source_to_videos, video_id_to_imgs, img_id_to_ann_id = defaultdict(list), defaultdict(list), defaultdict(list)
    [source_to_videos[video.get('source', 'default')].append(video) for video in gt['videos']]
    [video_id_to_imgs[img['video_id']].append(img) for img in gt['images']]
    [img_id_to_ann_id[ann['image_id']].append(ann) for ann in gt['annotations']]

    # Preds
    pred_img_id_to_anns = defaultdict(list)
    [pred_img_id_to_anns[ann['image_id']].append(ann) for ann in preds]

    eval_results = dict()
    for source, videos in source_to_videos.items():
        eval_save_dir = os.path.join(save_dir, source)
        os.makedirs(eval_save_dir, exist_ok=True)

        gt_coco = {'images': [], 'annotations': [], 'videos': videos, 'categories': gt['categories'], 'attributes': gt['attributes']}
        pred_list = []
        for video in videos: 
            for img in video_id_to_imgs[video['id']]: 
                if only_predicted_imgs and len(pred_img_id_to_anns[img['id']]) == 0: 
                    continue
                gt_coco['images'].append(img)
                for ann in img_id_to_ann_id[img['id']]: 
                    gt_coco['annotations'].append(ann)

                pred_list.extend(pred_img_id_to_anns[img['id']])
        
        with HiddenPrints():
            eval_result = evaluate(
                gt_coco, pred_list, eval_save_dir, 
                plot_examples=0, render_curves=0, verbose=0, class_weighted_metric=class_weighted_metric,
                with_velocity=with_velocity, tp_error_averaging=tp_error_averaging
            )

        eval_results[source] = {
            f'mAP': eval_result['mean_ap'],
            f'trans_err': eval_result['tp_errors']['trans_err'],
            f'scale_err': eval_result['tp_errors']['scale_err'],
            f'orient_err': eval_result['tp_errors']['orient_err'],
            f'vel_err': eval_result['tp_errors']['vel_err'],
            f'NDS': eval_result['nd_score'],
        }
        detailed_results = {
            source: eval_result
        }
    
    # combined_eval_results = {f'{source}_{name}': value for source, metric_dict in eval_results.items() for name, value in metric_dict.items() }
    
    return eval_results, detailed_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluate predictions on the MAGneT-3D cross-dataset benchmark.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('annotation_path', type=str, help='ground-truth annotation file (COCO style)')
    parser.add_argument('result_path', type=str, help='predictions (JSON list)')
    parser.add_argument('--save_dir', type=str, default='eval_output', help='where detailed metrics are written')
    parser.add_argument('--with_velocity', action='store_true', help='include the velocity term in NDS')
    parser.add_argument('--tp_error_averaging', choices=['recall', 'all_tp'], default='recall',
                        help='average the TP errors over recall levels (nuScenes) or over all true positives')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with open(args.result_path) as f:
        preds = json.load(f)
    results, _ = multi_dataset_evaluation(args.annotation_path, preds, args.save_dir, with_velocity=args.with_velocity,
                                          tp_error_averaging=args.tp_error_averaging)
    for source, m in results.items():
        print(f'{source}: mAP {100 * m["mAP"]:.2f}  NDS {100 * m["NDS"]:.2f}')
