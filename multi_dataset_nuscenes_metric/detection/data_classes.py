# nuScenes dev-kit.
# Code written by Oscar Beijbom, 2019.
# Modified for the MAGneT-3D cross-dataset benchmark.

import numpy as np
from typing import List, Tuple, Dict

from nuscenes.eval.common.data_classes import EvalBox, EvalBoxes, MetricData
from multi_dataset_nuscenes_metric.detection.constants import TP_METRICS
from nuscenes.eval.detection.data_classes import DetectionMetrics

from nuscenes.eval.common.utils import center_distance


class DetectionConfig:
    """ Data class that specifies the detection evaluation settings. """
    def __init__(self, class_names: List[str], dist_fcn: str, dist_ths: List[float], dist_th_tp: float,
                 min_recall: float, min_precision: float, max_boxes_per_sample: int, mean_ap_weight: int):

        assert dist_th_tp in dist_ths, "dist_th_tp must be in set of dist_ths."

        self.class_names = class_names
        self.dist_fcn = dist_fcn
        self.dist_ths = dist_ths
        self.dist_th_tp = dist_th_tp
        self.min_recall = min_recall
        self.min_precision = min_precision
        self.max_boxes_per_sample = max_boxes_per_sample
        self.mean_ap_weight = mean_ap_weight

    def serialize(self) -> dict:
        """ Serialize instance into json-friendly format. """
        return {
            'class_names': self.class_names,
            'dist_fcn': self.dist_fcn,
            'dist_ths': self.dist_ths,
            'dist_th_tp': self.dist_th_tp,
            'min_recall': self.min_recall,
            'min_precision': self.min_precision,
            'max_boxes_per_sample': self.max_boxes_per_sample,
            'mean_ap_weight': self.mean_ap_weight
        }

    @classmethod
    def deserialize(cls, class_names: List[str], content: dict):
        """ Initialize from serialized dictionary. """
        return cls(class_names, content['dist_fcn'], content['dist_ths'], content['dist_th_tp'], content['min_recall'],
                   content['min_precision'], content['max_boxes_per_sample'], content['mean_ap_weight'])

    @property
    def dist_fcn_callable(self):
        """ Return the distance function corresponding to the dist_fcn string. """
        if self.dist_fcn == 'center_distance':
            return center_distance
        else:
            raise Exception('Error: Unknown distance function %s!' % self.dist_fcn)


class DetectionMetricData(MetricData):
    """ This class holds accumulated and interpolated data required to calculate the detection metrics. """

    nelem = 101

    def __init__(self, recall: np.array, precision: np.array, confidence: np.array, trans_err: np.array, vel_err: np.array,
                 scale_err: np.array, orient_err: np.array):

        # Assert lengths.
        assert len(recall) == self.nelem
        assert len(precision) == self.nelem
        assert len(confidence) == self.nelem
        assert len(trans_err) == self.nelem
        assert len(vel_err) == self.nelem
        assert len(scale_err) == self.nelem
        assert len(orient_err) == self.nelem

        # Assert ordering.
        assert all(confidence == sorted(confidence, reverse=True))  # Confidences should be descending.
        assert all(recall == sorted(recall))  # Recalls should be ascending.

        # Set attributes explicitly to help IDEs figure out what is going on.
        self.recall = recall
        self.precision = precision
        self.confidence = confidence
        self.trans_err = trans_err
        self.vel_err = vel_err
        self.scale_err = scale_err
        self.orient_err = orient_err

    def __eq__(self, other):
        eq = True
        for key in self.serialize().keys():
            eq = eq and np.array_equal(getattr(self, key), getattr(other, key))
        return eq

    @property
    def max_recall_ind(self):
        """ Returns index of max recall achieved. """

        # Last instance of confidence > 0 is index of max achieved recall.
        non_zero = np.nonzero(self.confidence)[0]
        if len(non_zero) == 0:  # If there are no matches, all the confidence values will be zero.
            max_recall_ind = 0
        else:
            max_recall_ind = non_zero[-1]

        return max_recall_ind

    @property
    def max_recall(self):
        """ Returns max recall achieved. """

        return self.recall[self.max_recall_ind]

    def serialize(self):
        """ Serialize instance into json-friendly format. """
        return {
            'recall': self.recall.tolist(),
            'precision': self.precision.tolist(),
            'confidence': self.confidence.tolist(),
            'trans_err': self.trans_err.tolist(),
            'vel_err': self.vel_err.tolist(),
            'scale_err': self.scale_err.tolist(),
            'orient_err': self.orient_err.tolist()
        }

    @classmethod
    def deserialize(cls, content: dict):
        """ Initialize from serialized content. """
        return cls(recall=np.array(content['recall']), precision=np.array(content['precision']),
                   confidence=np.array(content['confidence']), trans_err=np.array(content['trans_err']),
                   vel_err=np.array(content['vel_err']), scale_err=np.array(content['scale_err']),
                   orient_err=np.array(content['orient_err']))

    @classmethod
    def no_predictions(cls):
        """ Returns a md instance corresponding to having no predictions. """
        return cls(recall=np.linspace(0, 1, cls.nelem), precision=np.zeros(cls.nelem), confidence=np.zeros(cls.nelem),
                   trans_err=np.ones(cls.nelem), vel_err=np.ones(cls.nelem), scale_err=np.ones(cls.nelem), orient_err=np.ones(cls.nelem))

    @classmethod
    def random_md(cls):
        """ Returns an md instance corresponding to a random results. """
        return cls(recall=np.linspace(0, 1, cls.nelem), precision=np.random.random(cls.nelem),
                   confidence=np.linspace(0, 1, cls.nelem)[::-1], trans_err=np.random.random(cls.nelem), vel_err=np.random.random(cls.nelem),
                   scale_err=np.random.random(cls.nelem), orient_err=np.random.random(cls.nelem))


class DetectionBox(EvalBox):
    """ Data class used during detection evaluation. Can be a prediction or ground truth."""
    def __init__(
        self,
        sample_token: str = "",
        translation: Tuple[float, float, float] = (0, 0, 0),
        size: Tuple[float, float, float] = (0, 0, 0),
        rotation: Tuple[float, float, float, float] = (0, 0, 0, 0),
        velocity: Tuple[float, float] = (0, 0),
        ego_translation: [float, float, float] = (0, 0, 0),  # Translation to ego vehicle in meters.
        num_pts: int = -1,  # Nbr. LIDAR or RADAR inside the box. Only for gt boxes.
        detection_name: str = 'car',  # The class name used in the detection challenge.
        detection_score: float = -1.0,  # GT samples do not have a score.
        attribute_name: str = ''):  # Box attribute. Each box can have at most 1 attribute.

        super().__init__(sample_token, translation, size, rotation, velocity, ego_translation, num_pts)

        assert detection_name is not None, 'Error: detection_name cannot be empty!'

        assert type(detection_score) == float, 'Error: detection_score must be a float!'
        assert not np.any(np.isnan(detection_score)), 'Error: detection_score may not be NaN!'

        # Assign.
        self.detection_name = detection_name
        self.detection_score = detection_score
        self.attribute_name = attribute_name

    def serialize(self) -> dict:
        raise NotImplementedError

    @classmethod
    def deserialize(cls, content: dict):
        raise NotImplementedError


class WeightedDetectionMetrics(DetectionMetrics):
    """
    Weighted averages with respect to number of labels of the given class.
    Helpful for unbalanced dataset
    """
    def __init__(self, cfg: DetectionConfig, class_weights: Dict[str, float], with_velocity: bool = False):
        super().__init__(cfg)
        self.class_weights = class_weights
        self.with_velocity = with_velocity

    @classmethod
    def from_boxes(cls: DetectionMetrics, cfg: DetectionConfig, boxes: EvalBoxes, class_weighted_metric: bool = True,
                   with_velocity: bool = False):
        class_weights = {}
        for class_name in cfg.class_names:
            class_weights[class_name] = len([b for b in boxes.all if b.detection_name == class_name]) if class_weighted_metric else 1
        return cls(cfg, class_weights, with_velocity)

    @property
    def nd_score(self) -> float:
        """ NDS: weighted mean of mAP and the true positive scores, without velocity unless ``with_velocity``. """
        scores = [v for k, v in self.tp_scores.items() if self.with_velocity or k != 'vel_err']
        return (self.cfg.mean_ap_weight * self.mean_ap + sum(scores)) / (self.cfg.mean_ap_weight + len(scores))

    @property
    def mean_ap(self) -> float:
        """ Calculates the weighted mean AP by averaging over distance thresholds and classes. """
        norm = sum(w for w in self.class_weights.values())
        return sum(v * self.class_weights[class_name] for (class_name, v) in self.mean_dist_aps.items()) / norm

    @property
    def tp_errors(self) -> Dict[str, float]:
        """ Calculates the mean true positive error across all classes for each metric. """
        weights = np.array([self.class_weights[class_name] for class_name in self.cfg.class_names])
        errors = {}
        for metric_name in TP_METRICS:
            class_errors = np.array(
                [self.get_label_tp(detection_name, metric_name) for detection_name in self.cfg.class_names])
            not_nan_idxs = ~np.isnan(class_errors)
            errors[metric_name] = np.nansum(class_errors * weights) / np.sum(weights[not_nan_idxs])
        return errors

    @property
    def tp_scores(self) -> Dict[str, float]:
        scores = {}
        tp_errors = self.tp_errors
        for metric_name in TP_METRICS:

            # We convert the true positive errors to "scores" by 1-error.
            score = 1.0 - tp_errors[metric_name]

            # Some of the true positive errors are unbounded, so we bound the scores to min 0.
            score = max(0.0, score)

            scores[metric_name] = score

        return scores

    def serialize(self):
        ret = super().serialize()
        ret['class_weights'] = self.class_weights
        return ret
