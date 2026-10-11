# nuScenes dev-kit.
# Code written by Holger Caesar, 2019.
# Modified for the MAGneT-3D cross-dataset benchmark.

import json
import os
import pkg_resources

import pycocotools.coco as coco

from multi_dataset_nuscenes_metric.detection.data_classes import DetectionConfig


def config_factory(annotation: coco.COCO, cfg_cls) -> DetectionConfig:
    """
    Creates a DetectionConfig from the nuScenes detection config 'detection_cvpr_2019.json' of the installed
    devkit, with the class names of the annotation file.
    :param annotation: The annotation file in COCO format.
    :param cfg_cls: DetectionConfig.
    :return: DetectionConfig instance.
    """
    assert cfg_cls == DetectionConfig, 'Error: Invalid config type {}'.format(cfg_cls.__name__)
    cfg_path = pkg_resources.resource_filename(
        'nuscenes', os.path.join('eval', 'detection', 'configs', 'detection_cvpr_2019.json'))
    assert os.path.exists(cfg_path), 'Config path {} does not exist'.format(cfg_path)

    # Extract class names.
    class_names = [cls['name'] for cls in annotation.dataset['categories']]

    # Load config file and deserialize it.
    with open(cfg_path, 'r') as f:
        data = json.load(f)
    return DetectionConfig.deserialize(class_names, data)
