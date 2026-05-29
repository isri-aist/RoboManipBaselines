# Using a tactile sensor in MuJoCo

## Install
Install the [MujocoTactileSensorPlugin](https://github.com/isri-aist/MujocoTactileSensorPlugin).

> [!NOTE]  
> For installation instructions, please refer to [this guide](https://github.com/isri-aist/MujocoTactileSensorPlugin/tree/main?tab=readme-ov-file#install).  
> The MujocoTactileSensorPlugin supports two installation modes: as a standalone project or as a ROS package.  
> **Please install it as a standalone project for use with RoboManipBaselines.**

After installation, copy the plugin library files into the `plugin` directory of your MuJoCo Python module:
```console
$ cp <CMAKE_INSTALL_PREFIX>/lib/libTactileSensor.so <path_to_venv>/lib/python3.10/site-packages/mujoco/plugin
$ cp <CMAKE_INSTALL_PREFIX>/lib/libTactileSensorPlugin.so <path_to_venv>/lib/python3.10/site-packages/mujoco/plugin
```

## How to run
Add the following line to your MuJoCo XML file:
```xml
  <include file="../../robots/ur5e/ur5e_tactile_sensor_config.xml"/>
```

As an example, uncomment the following line in [env_ur5e_pick.xml](../robo_manip_baselines/envs/assets/mujoco/envs/ur5e/env_ur5e_pick.xml):
```xml
  <!-- To enable the tactile sensor, comment in the following line. -->
  <!-- <include file="../../robots/ur5e/ur5e_tactile_sensor_config.xml"/> -->
```

Once enabled, you can launch teleoperation as usual and the tactile sensors will be activated:
```console
$ uv run python ./bin/Teleop.py MujocoUR5ePick
```
The measurements from each tactile sensor are automatically stored in the RMB data using the sensor name as the data key.

To visualize the tactile measurements, enable the `--plot_tactile` option:
```console
$ uv run python ./bin/Teleop.py MujocoUR5ePick --plot_tactile
```

https://github.com/user-attachments/assets/2c2719da-db1e-42b8-bc2d-4efc207813be
