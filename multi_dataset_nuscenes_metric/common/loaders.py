# nuScenes dev-kit.
# Code written by Oscar Beijbom, 2019.
# Modified for the MAGneT-3D cross-dataset benchmark.

import os
from math import sqrt
from typing import Dict, List, Union

import numpy as np
from scipy.spatial.transform import Rotation
from pyquaternion import Quaternion
import pycocotools.coco as coco

from nuscenes.eval.common.data_classes import EvalBoxes

from multi_dataset_nuscenes_metric.detection.data_classes import DetectionBox


def matrix_to_quaternion(matrix: np.ndarray) -> Quaternion:
    """Same as ``Quaternion(matrix=matrix)`` (pyquaternion's trace method) but
    without its orthogonality checks, which dominate the loading time. The
    matrices passed here come from ``Rotation.from_rotvec`` and are orthogonal
    by construction."""
    m = matrix.conj().transpose()
    if m[2, 2] < 0:
        if m[0, 0] > m[1, 1]:
            t = 1 + m[0, 0] - m[1, 1] - m[2, 2]
            q = [m[1, 2] - m[2, 1], t, m[0, 1] + m[1, 0], m[2, 0] + m[0, 2]]
        else:
            t = 1 - m[0, 0] + m[1, 1] - m[2, 2]
            q = [m[2, 0] - m[0, 2], m[0, 1] + m[1, 0], t, m[1, 2] + m[2, 1]]
    else:
        if m[0, 0] < -m[1, 1]:
            t = 1 - m[0, 0] - m[1, 1] + m[2, 2]
            q = [m[0, 1] - m[1, 0], m[2, 0] + m[0, 2], m[1, 2] + m[2, 1], t]
        else:
            t = 1 + m[0, 0] + m[1, 1] + m[2, 2]
            q = [t, m[1, 2] - m[2, 1], m[2, 0] - m[0, 2], m[0, 1] - m[1, 0]]
    q = np.array(q).astype('float64')
    q *= 0.5 / sqrt(t)
    return Quaternion(array=q)


def convert_coco_item_to_eval_box(sample_token: str, item: Dict, cls_id_to_name: Dict[int, str],
                                  att_id_to_name: Dict[int, str], box_cls) -> DetectionBox:
    """
    Converts an item given in COCO format into a DetectionBox.
    """
    size = (item['dim'][1], item['dim'][2], item['dim'][0])  # w, l, h
    rotation_matrix = Rotation.from_rotvec(item['rotation']).as_matrix()

    translation = np.array(item['translation']) + 0.5 * size[2] * rotation_matrix[:, 2]  # center of the box [x, y, z]

    rotation = tuple(matrix_to_quaternion(rotation_matrix))
    velocity = item['velocity'][:2]
    detection_name = cls_id_to_name[item['category_id']]
    attribute_name = att_id_to_name.get(item.get('attribute_id')) or ''

    score = item['score'] if 'score' in item else -1.0
    assert box_cls == DetectionBox, "Error: Invalid box type {}".format(box_cls.__name__)
    return DetectionBox(sample_token, translation, size, rotation, velocity, detection_name=detection_name,
                        detection_score=score, attribute_name=attribute_name)


def load_boxes(coco: coco.COCO, classes: List[Dict[str, Union[str, int]]], attributes: List[Dict[str, Union[str, int]]],
               max_boxes_per_sample: int, box_cls, gt: bool, verbose: bool = False) -> EvalBoxes:
    """
    Loads EvalBoxes from data given in COCO format.
    """
    boxes = EvalBoxes()
    cls_id_to_name = {cls['id']: cls['name'] for cls in classes}
    cls_id_to_max_eval_distance = {cls['id']: cls.get('max_eval_distance') for cls in classes}
    att_id_to_name = {att['id']: att['name'] for att in attributes}
    filtered_count = 0
    total_count = 0

    for img_info in coco.imgs.values():
        data = coco.loadAnns(coco.getAnnIds(img_info['id']))
        sample_token = os.path.splitext(img_info['file_name'])[0]

        filtered_boxes = []
        for item in data:
            if item['category_id'] not in cls_id_to_name:
                assert False, "Error: Unknown class id {} found!".format(item['category_id'])
                continue
            total_count += 1
            max_eval_distance = cls_id_to_max_eval_distance[item['category_id']]
            if max_eval_distance is not None:
                # Ground plane distance from the camera (ignoring height), like the nuScenes class ranges.
                assert 'translation_in_camera_frame' in item, \
                    "Error: 'translation_in_camera_frame' is required for the range filter."
                x_cam, y_cam, z_cam = item['translation_in_camera_frame']
                distance = np.sqrt(x_cam**2 + z_cam**2)
                if distance > max_eval_distance:
                    filtered_count += 1
                    continue

            filtered_boxes.append(
                convert_coco_item_to_eval_box(sample_token, item, cls_id_to_name, att_id_to_name, box_cls))

        boxes.add_boxes(sample_token=sample_token, boxes=filtered_boxes)

    if verbose:
        print("Loaded {} {} boxes from {} samples.".format(len(boxes.all), ['PRED', 'GT'][gt],
                                                           len(boxes.sample_tokens)))

    # Check that each sample has no more than x boxes.
    for sample_token in boxes.sample_tokens:
        assert len(boxes.boxes[sample_token]) <= max_boxes_per_sample, \
            "Error: Only <= %d boxes per sample allowed!" % max_boxes_per_sample

    return boxes
