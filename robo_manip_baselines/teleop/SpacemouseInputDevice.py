import numpy as np
import pinocchio as pin

from .InputDeviceBase import InputDeviceBase


class SpacemouseInputDevice(InputDeviceBase):
    """Spacemouse for teleoperation input device."""

    def __init__(
        self,
        arm_manager,
        pos_scale=1e-2,
        rpy_scale=5e-3,
        gripper_scale=5.0,
        device_params={},
    ):
        super().__init__()

        self.arm_manager = arm_manager
        self.pos_scale = pos_scale
        self.rpy_scale = rpy_scale
        self.gripper_scale = gripper_scale
        self.device_params = device_params

    def connect(self):
        if self.connected:
            return

        self.connected = True

        import pyspacemouse

        self.spacemouse = pyspacemouse.open(**self.device_params)

    def read(self):
        if not self.connected:
            raise RuntimeError(f"[{self.__class__.__name__}] Device is not connected.")

        # Empirically, you can call read repeatedly to get the latest device state
        for i in range(10):
            self.state = self.spacemouse.read()

    def set_command_data(self):
        # Set arm command
        delta_pos = self.pos_scale * np.array(
            [
                -1.0 * self.state.y,
                self.state.x,
                self.state.z,
            ]
        )
        delta_rpy = self.rpy_scale * np.array(
            [
                -1.0 * self.state.roll,
                -1.0 * self.state.pitch,
                -2.0 * self.state.yaw,
            ]
        )

        target_se3 = self.arm_manager.target_se3.copy()
        target_se3.translation += delta_pos
        target_se3.rotation = pin.rpy.rpyToMatrix(*delta_rpy) @ target_se3.rotation

        self.arm_manager.set_command_eef_pose(target_se3)

        # Set gripper command
        gripper_joint_pos = self.arm_manager.get_command_gripper_joint_pos().copy()

        if self.state.buttons[0] > 0 and self.state.buttons[-1] <= 0:
            gripper_joint_pos += self.gripper_scale
        elif self.state.buttons[-1] > 0 and self.state.buttons[0] <= 0:
            gripper_joint_pos -= self.gripper_scale

        self.arm_manager.set_command_gripper_joint_pos(gripper_joint_pos)

    def is_active(self) -> bool:
        if self.state is None:
            return False
        # Check for any movement, rotation, or button press
        return (
            abs(self.state.x) > 1e-6
            or abs(self.state.y) > 1e-6
            or abs(self.state.z) > 1e-6
            or abs(self.state.roll) > 1e-6
            or abs(self.state.pitch) > 1e-6
            or abs(self.state.yaw) > 1e-6
            or any(self.state.buttons)
        )
