# nuScenes dev-kit.
# Code written by Holger Caesar, Varun Bankiti, and Alex Lang, 2019.
# Modified for the MAGneT-3D cross-dataset benchmark.

import os
from typing import Any

import numpy as np
from pyquaternion import Quaternion
from matplotlib import pyplot as plt

from nuscenes.utils.data_classes import Box
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.common.render import setup_axis
from nuscenes.eval.detection.constants import TP_METRICS_UNITS, PRETTY_TP_METRICS
from multi_dataset_nuscenes_metric.detection.constants import TP_METRICS
from nuscenes.eval.detection.data_classes import DetectionMetrics, DetectionMetricData, DetectionMetricDataList

Axis = Any


def visualize_sample(sample_token: str, gt_boxes: EvalBoxes, pred_boxes: EvalBoxes, conf_th: float = 0.15,
                     verbose: bool = True, savepath: str = None) -> None:
    """
    Visualizes a sample from top and side view with annotations and detection results.
    :param sample_token: The sample token.
    :param gt_boxes: Ground truth boxes grouped by sample.
    :param pred_boxes: Prediction grouped by sample.
    :param conf_th: The confidence threshold used to filter negatives.
    :param verbose: Whether to print to stdout.
    :param savepath: If given, saves the the rendering here instead of displaying.
    """
    # Init axes.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))
    ax1.set_aspect('equal')
    ax2.set_aspect('equal')

    # Define projections.
    top_view = np.eye(3)
    side_view = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])

    # Show GT boxes.
    for gt_box in gt_boxes[sample_token]:
        box = Box(gt_box.translation, gt_box.size, Quaternion(gt_box.rotation))
        box.render(ax1, view=top_view, colors=('g', 'g', 'g'), linewidth=2)
        box.render(ax2, view=side_view, colors=('g', 'g', 'g'), linewidth=2)

    # Show PRED boxes.
    for pred_box in pred_boxes[sample_token]:
        # Show only predictions with a high score.
        assert not np.isnan(pred_box.detection_score), 'Error: Box score cannot be NaN!'
        if pred_box.detection_score >= conf_th:
            box = Box(pred_box.translation, pred_box.size, Quaternion(pred_box.rotation))
            box.render(ax1, view=top_view, colors=('b', 'b', 'b'), linewidth=1)
            box.render(ax2, view=side_view, colors=('b', 'b', 'b'), linewidth=1)

    # Set axis labels.
    ax1.set_xlabel('x [m]')
    ax1.set_ylabel('y [m]')
    ax2.set_xlabel('x [m]')
    ax2.set_ylabel('z [m]')

    # Show / save plot.
    if verbose:
        print('Rendering sample token %s' % sample_token)
    fig.suptitle(sample_token)
    if savepath is not None:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        plt.savefig(savepath)
        plt.close()
    else:
        plt.show()


def class_pr_curve(md_list: DetectionMetricDataList, metrics: DetectionMetrics, detection_name: str,
                   min_precision: float, min_recall: float, savepath: str = None, ax: Axis = None) -> None:
    """
    Plot a precision recall curve for the specified class.
    :param md_list: DetectionMetricDataList instance.
    :param metrics: DetectionMetrics instance.
    :param detection_name: The detection class.
    :param min_precision:
    :param min_recall: Minimum recall value.
    :param savepath: If given, saves the the rendering here instead of displaying.
    :param ax: Axes onto which to render.
    """
    # Prepare axis.
    if ax is None:
        ax = setup_axis(title=detection_name, xlabel='Recall', ylabel='Precision', xlim=1, ylim=1,
                        min_precision=min_precision, min_recall=min_recall)

    # Get recall vs precision values of given class for each distance threshold.
    data = md_list.get_class_data(detection_name)

    # Plot the recall vs. precision curve for each distance threshold.
    for md, dist_th in data:
        md: DetectionMetricData
        ap = metrics.get_label_ap(detection_name, dist_th)
        ax.plot(md.recall, md.precision, label='Dist. : {}, AP: {:.1f}'.format(dist_th, ap * 100))

    ax.legend(loc='best')
    if savepath is not None:
        plt.savefig(savepath)
        plt.close()


def class_tp_curve(md_list: DetectionMetricDataList, metrics: DetectionMetrics, detection_name: str, min_recall: float,
                   dist_th_tp: float, savepath: str = None, ax: Axis = None) -> None:
    """
    Plot the true positive curve for the specified class.
    :param md_list: DetectionMetricDataList instance.
    :param metrics: DetectionMetrics instance.
    :param detection_name:
    :param min_recall: Minimum recall value.
    :param dist_th_tp: The distance threshold used to determine matches.
    :param savepath: If given, saves the the rendering here instead of displaying.
    :param ax: Axes onto which to render.
    """
    # Get metric data for given detection class with tp distance threshold.
    md = md_list[(detection_name, dist_th_tp)]
    min_recall_ind = round(100 * min_recall)
    if min_recall_ind <= md.max_recall_ind:
        # For traffic_cone and barrier only a subset of the metrics are plotted.
        rel_metrics = [m for m in TP_METRICS if not np.isnan(metrics.get_label_tp(detection_name, m))]
        ylimit = max([max(getattr(md, metric)[min_recall_ind:md.max_recall_ind + 1]) for metric in rel_metrics]) * 1.1
    else:
        ylimit = 1.0

    # Prepare axis.
    if ax is None:
        ax = setup_axis(title=detection_name, xlabel='Recall', ylabel='Error', xlim=1, min_recall=min_recall)
    ax.set_ylim(0, ylimit)

    # Plot the recall vs. error curve for each tp metric.
    for metric in TP_METRICS:
        tp = metrics.get_label_tp(detection_name, metric)

        # Plot only if we have valid data.
        if tp is not np.nan and min_recall_ind <= md.max_recall_ind:
            recall, error = md.recall[:md.max_recall_ind + 1], getattr(md, metric)[:md.max_recall_ind + 1]
        else:
            recall, error = [], []

        # Change legend based on tp value
        if tp is np.nan:
            label = '{}: n/a'.format(PRETTY_TP_METRICS[metric])
        elif min_recall_ind > md.max_recall_ind:
            label = '{}: nan'.format(PRETTY_TP_METRICS[metric])
        else:
            label = '{}: {:.2f} ({})'.format(PRETTY_TP_METRICS[metric], tp, TP_METRICS_UNITS[metric])
        ax.plot(recall, error, label=label)
    ax.axvline(x=md.max_recall, linestyle='-.', color=(0, 0, 0, 0.3))
    ax.legend(loc='best')

    if savepath is not None:
        plt.savefig(savepath)
        plt.close()


def dist_pr_curve(md_list: DetectionMetricDataList, metrics: DetectionMetrics, dist_th: float, min_precision: float,
                  min_recall: float, savepath: str = None) -> None:
    """
    Plot the PR curves for different distance thresholds.
    :param md_list: DetectionMetricDataList instance.
    :param metrics: DetectionMetrics instance.
    :param dist_th: Distance threshold for matching.
    :param min_precision: Minimum precision value.
    :param min_recall: Minimum recall value.
    :param savepath: If given, saves the the rendering here instead of displaying.
    """
    # Prepare axis.
    fig, (ax, lax) = plt.subplots(ncols=2, gridspec_kw={"width_ratios": [4, 1]}, figsize=(7.5, 5))
    ax = setup_axis(xlabel='Recall', ylabel='Precision', xlim=1, ylim=1, min_precision=min_precision,
                    min_recall=min_recall, ax=ax)

    # Plot the recall vs. precision curve for each detection class.
    data = md_list.get_dist_data(dist_th)
    for md, detection_name in data:
        md = md_list[(detection_name, dist_th)]
        ap = metrics.get_label_ap(detection_name, dist_th)
        ax.plot(md.recall, md.precision, label='{}: {:.1f}%'.format(detection_name, ap * 100))
    hx, lx = ax.get_legend_handles_labels()
    lax.legend(hx, lx, borderaxespad=0)
    lax.axis("off")
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath)
        plt.close()


def summary_plot(md_list: DetectionMetricDataList, metrics: DetectionMetrics, min_precision: float, min_recall: float,
                 dist_th_tp: float, savepath: str = None) -> None:
    """
    Creates a summary plot with PR and TP curves for each class.
    :param md_list: DetectionMetricDataList instance.
    :param metrics: DetectionMetrics instance.
    :param min_precision: Minimum precision value.
    :param min_recall: Minimum recall value.
    :param dist_th_tp: The distance threshold used to determine matches.
    :param savepath: If given, saves the the rendering here instead of displaying.
    """
    n_classes = len(metrics.cfg.class_names)
    _, axes = plt.subplots(nrows=n_classes, ncols=2, figsize=(15, 5 * n_classes), squeeze=False)
    for ind, detection_name in enumerate(metrics.cfg.class_names):
        title1, title2 = ('Recall vs Precision', 'Recall vs Error') if ind == 0 else (None, None)

        ax1 = setup_axis(xlim=1, ylim=1, title=title1, min_precision=min_precision, min_recall=min_recall, ax=axes[ind,
                                                                                                                   0])
        ax1.set_ylabel('{} \n \n Precision'.format(detection_name), size=20)

        ax2 = setup_axis(xlim=1, title=title2, min_recall=min_recall, ax=axes[ind, 1])
        if ind == n_classes - 1:
            ax1.set_xlabel('Recall', size=20)
            ax2.set_xlabel('Recall', size=20)

        class_pr_curve(md_list, metrics, detection_name, min_precision, min_recall, ax=ax1)
        class_tp_curve(md_list, metrics, detection_name, min_recall, dist_th_tp=dist_th_tp, ax=ax2)

    plt.tight_layout()

    if savepath is not None:
        plt.savefig(savepath)
        plt.close()
