# How to setup multiple SpaceMouse devices

For high-degree-of-freedom robots such as dual-arm manipulators or mobile manipulators, it is possible to use multiple SpaceMouse devices for teleoperation.
To enable such teleoperation, specify the device configuration file using the `--input_device_config` argument in `Teleop.py` as shown below.
```console
# Example for a dual-arm manipulator
$ uv run python ./bin/Teleop.py MujocoUR5eDualCable --input_device_config ./teleop/configs/SpaceMouseDual.yaml

# Example for a mobile manipulator
$ uv run python ./bin/Teleop.py MujocoHsrTidyup --input_device_config ./teleop/configs/SpaceMouseDual.yaml
```

The contents of `SpaceMouseDual.yaml` are as follows.
```yaml
0:
  device_params:
    path: /dev/hidraw6
1:
  device_params:
    path: /dev/hidraw7
```

The keys `0` and `1` represent different parts of the robot being operated (e.g., left and right arms, or mobile base and arm).
The `path` entry specifies the device file path corresponding to each SpaceMouse.
You can check the device file path using the following command.
```console
$ sudo dmesg | grep "3Dconnexion SpaceMouse" | grep hidraw
```

For example, if the output includes a line like the following, the device file path is `/dev/hidraw6`.
```console
[...] hid-generic ...: input,hidraw6: USB HID v1.11 Multi-Axis Controller [3Dconnexion SpaceMouse Wireless BT] on usb-...
```
