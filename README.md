# MAGneT-3D cross-dataset benchmark

Evaluation code of the cross-dataset benchmark for monocular temporal 3D detection introduced in
[MAGneT-3D](https://github.com/mo-sameh/MAGneT-3D). Models are trained on one of nuScenes, Waymo or
Lyft and evaluated on the validation sets of nuScenes, Waymo, Lyft and ONCE (front camera), using
the shared taxonomy `vehicle`, `pedestrian`, `bicycle`.

## Metric

The metric follows the nuScenes detection protocol (center-distance matching on the ground plane,
per-class evaluation ranges) on COCO-style annotation files. We report mAP and the nuScenes
detection score without the velocity term,

    NDS = (5 mAP + s_trans + s_scale + s_orient) / 8,    s_x = 1 - min(1, x_err)

(`--with_velocity` / `with_velocity=True` adds the velocity term, divided by 9).

```bash
pip install git+https://github.com/mo-sameh/MAGneT-3D-Benchmark.git
python -m multi_dataset_nuscenes_metric.detection.evaluate <annotation.json> <results.json>
```

or from Python:

```python
from multi_dataset_nuscenes_metric.detection.evaluate import multi_dataset_evaluation
metrics, _ = multi_dataset_evaluation('nuscenes_val.json', predictions, 'eval_output')  # {source: {'mAP', 'NDS', ...}}
```

Predictions are a list of boxes in the global frame:
`{"image_id", "category_id", "score", "translation" (bottom center, m), "dim" (h, w, l),
"rotation" (rotation vector), "velocity", "translation_in_camera_frame"}`. `category_id` is the id of
the class in the `categories` of the annotation file; map the classes by name, the ids differ between
datasets.

## Data

The four datasets are converted to the common annotation format with
[time-dataset-converter](https://gitlab.lrz.de/johannes-tum/time-dataset-converter).

This code is based on the [nuScenes devkit](https://github.com/nutonomy/nuscenes-devkit)
(Apache-2.0); see [NOTICE](NOTICE).
