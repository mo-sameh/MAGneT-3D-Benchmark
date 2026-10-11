from setuptools import find_packages, setup

setup(
    name='multi_dataset_nuscenes_metric',
    version='0.1.0',
    description='nuScenes-style detection metric for the MAGneT-3D cross-dataset benchmark',
    packages=find_packages(include=['multi_dataset_nuscenes_metric', 'multi_dataset_nuscenes_metric.*']),
    license='Apache-2.0',
    install_requires=['nuscenes-devkit==1.1.11', 'pycocotools', 'pyquaternion', 'scipy', 'numpy'],
)
