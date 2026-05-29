# Data collection by teleoperation

A full list of teleoperation environments can be found in [the environment catalog](../../doc/environment_catalog.md).

## Record demonstration data by teleoperation
You need [SpaceMouse Wireless](https://3dconnexion.com/us/product/spacemouse-wireless) for teleoperation.
Connect SpaceMouse to your PC before launching the script.

Start up the teleoperation environment by the following command:
```console
# Go to the top directory of this repository
$ cd robo_manip_baselines
$ uv run python ./bin/Teleop.py MujocoUR5eCable
```

> [!NOTE]  
> If SpaceMouse is not detected or you get `Failed to open device`, apply the following fix:
>
> ```bash
> # Install HID dependency
> $ sudo apt-get install libhidapi-dev
>
> # Allow hidraw device access for SpaceMouse
> $ sudo tee /etc/udev/rules.d/99-hidraw-permissions.rules <<EOF
> KERNEL=="hidraw*", SUBSYSTEM=="hidraw", MODE="0664", GROUP="plugdev"
> EOF
>
> # Reload udev rules
> $ sudo udevadm control --reload-rules
> $ sudo udevadm trigger
>
> # Add user to plugdev group
> $ sudo usermod -aG plugdev $USER
> $ newgrp plugdev
>
> # Verify device access
> $ ls -l /dev/hidraw*
> $ groups
> ```

https://github.com/user-attachments/assets/59736023-a7f7-4aca-a860-176db84579f7

If you want to use a keyboard as a teleoperation input device:
```console
$ uv run python ./bin/Teleop.py MujocoUR5eCable --input_device keyboard
```

If you want to use [GELLO](https://wuphilipp.github.io/gello_site) as a teleoperation input device:
```console
$ uv run python ./bin/Teleop.py MujocoUR5eCable --input_device gello
```

If you want to use [HTC Vive Controller](https://www.vive.com/us/accessory/controller2018) as a teleoperation input device:
```console
$ uv run python ./bin/Teleop.py MujocoUR5eCable --input_device vive --input_device_config ./teleop/configs/Vive.yaml
```

If you want to save the data in [RmbData-SingleHDF5 (`.hdf5`) format instead of RmbData-Compact (`.rmb`)](../../doc/rmb_data_format.md):
```console
$ uv run python ./bin/Teleop.py MujocoUR5eCable --file_format hdf5
```

If you want to collect data only in a simulation environment with limited world indices (for example, only 0 and 5), add the following option:
```console
$ uv run python ./bin/Teleop.py MujocoUR5eCable --world_idx_list 0 5
```

To plot the point cloud, add the following option:
```console
$ uv run python ./bin/Teleop.py MujocoUR5eCable --plot_pointcloud
```

> [!NOTE]
> If you cannot zoom the point cloud view by right-clicking, try changing the matplotlib version: `pip install matplotlib=="3.6.1"`.

To replay the teleoperation motion of the log, add the following option:
```console
$ uv run python ./bin/Teleop.py MujocoUR5eCable --replay_log ./dataset/MujocoUR5eCable_<date_suffix>/MujocoUR5eCable_env0_000.rmb
```
