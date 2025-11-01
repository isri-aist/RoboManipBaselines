# Data preprocessing for policies using 3D point clouds

For policies that use 3D point clouds, apply the following preprocessing steps to your teleoperation dataset.

Decide the parameters of the bounding box for cropping point clouds:
```console
# Go to the top directory of this repository
$ cd robo_manip_baselines
$ python ./misc/VisualizePointCloud.py ./dataset/<dataset_name>/<rmb_file_name>
```
After adjusting the bounding box position and size, as well as the point cloud orientation, using the keyboard or a 3D mouse, press the "P" key to print the bounding box parameters.

Generate and store point clouds from RGB and depth images:
```console
# Go to the top directory of this repository
$ cd robo_manip_baselines
$ python ./misc/AddPointCloudToRmbData.py ./dataset/<dataset_name> --min_bound <x, y, z> --max_bound <x, y, z> --rpy_angle <roll, pitch, yaw>
```
For bounding box parameters such as `min_bound`, you can use the results adjusted in `VisualizePointCloud.py`.

> [!TIP]
> You can also download and use datasets containing point clouds from [here](../../../doc/dataset_list.md#ur5e--demo-30).
