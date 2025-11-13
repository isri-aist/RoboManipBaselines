import threading

import numpy as np
import pinocchio as pin

from .InputDeviceBase import InputDeviceBase


class KeyboardInputDevice(InputDeviceBase):
    """Keyboard for teleoperation input device."""

    def __init__(
        self,
        arm_manager,
        pos_scale=1e-2,
        rpy_scale=5e-2,
        gripper_scale=50.0,
    ):
        super().__init__()

        self.arm_manager = arm_manager
        self.pos_scale = pos_scale
        self.rpy_scale = rpy_scale
        self.gripper_scale = gripper_scale

        self.state = {
            # position control keys
            "w": False,
            "s": False,
            "a": False,
            "d": False,
            "q": False,
            "e": False,
            #  rotation control keys
            "i": False,
            "k": False,
            "j": False,
            "l": False,
            "u": False,
            "o": False,
            # gripper control keys
            "z": False,  # close gripper
            "x": False,  # open gripper
        }

        self.listener = None
        self.listener_thread = None

    def connect(self):
        if self.connected:
            return

        from pynput import keyboard

        self.listener = keyboard.Listener(
            on_press=self._on_press, on_release=self._on_release
        )

        # keyboard listener another thread
        self.listener_thread = threading.Thread(target=self._start_listener)
        self.listener_thread.daemon = True
        self.listener_thread.start()

        self.connected = True
        print(f"[{self.__class__.__name__}] Connected.")
        print(f"""[{self.__class__.__name__}] Key Bindings:
  - WASD : XY movement
  - QE   : Z-axis movement
  - IJKL : Roll and Pitch rotation
  - UO  : Yaw rotation
  - Z/X  : Gripper open/close""")

    def _start_listener(self):
        self.listener.start()
        self.listener.join()

    def _on_press(self, key):
        try:
            k = key.char.lower()
            if k in self.state:
                self.state[k] = True
        except AttributeError:
            pass

    def _on_release(self, key):
        try:
            k = key.char.lower()
            if k in self.state:
                self.state[k] = False
        except AttributeError:
            pass

    def read(self):
        if not self.connected:
            raise RuntimeError(f"[{self.__class__.__name__}] Device is not connected.")

    def set_command_data(self):
        delta_pos = np.zeros(3)

        # X-axis
        if self.state["w"]:
            delta_pos[0] += self.pos_scale
        if self.state["s"]:
            delta_pos[0] -= self.pos_scale

        # Y-axis
        if self.state["a"]:
            delta_pos[1] += self.pos_scale
        if self.state["d"]:
            delta_pos[1] -= self.pos_scale

        # Z-axis
        if self.state["q"]:
            delta_pos[2] += self.pos_scale
        if self.state["e"]:
            delta_pos[2] -= self.pos_scale

        delta_rpy = np.zeros(3)

        # Roll
        if self.state["j"]:
            delta_rpy[0] -= self.rpy_scale
        if self.state["l"]:
            delta_rpy[0] += self.rpy_scale

        # Pitch
        if self.state["i"]:
            delta_rpy[1] += self.rpy_scale
        if self.state["k"]:
            delta_rpy[1] -= self.rpy_scale

        # Yaw
        if self.state["u"]:
            delta_rpy[2] += self.rpy_scale * 2.0
        if self.state["o"]:
            delta_rpy[2] -= self.rpy_scale * 2.0

        target_se3 = self.arm_manager.target_se3.copy()
        target_se3.translation += delta_pos
        target_se3.rotation = pin.rpy.rpyToMatrix(*delta_rpy) @ target_se3.rotation

        self.arm_manager.set_command_eef_pose(target_se3)

        # Set gripper command
        gripper_joint_pos = self.arm_manager.get_command_gripper_joint_pos().copy()

        if self.state["z"] and not self.state["x"]:
            gripper_joint_pos += self.gripper_scale
        elif self.state["x"] and not self.state["z"]:
            gripper_joint_pos -= self.gripper_scale

        self.arm_manager.set_command_gripper_joint_pos(gripper_joint_pos)

    def disconnect(self):
        if self.connected:
            if self.listener:
                self.listener.stop()
            self.connected = False
            print(f"[{self.__class__.__name__}] Disconnected.")

    def is_active(self) -> bool:
        if self.state is None:
            return False
        # Check if any of the control keys are currently pressed
        return any(self.state.values())


class AzertyKeyboardInputDevice(KeyboardInputDevice):
    """AZERTY Keyboard for teleoperation input device."""

    def __init__(
        self,
        arm_manager,
        pos_scale=1e-2,
        rpy_scale=5e-2,
        gripper_scale=50.0,
    ):
        super().__init__(arm_manager, pos_scale, rpy_scale, gripper_scale)

        self.state = {
            # position control keys (AZERTY)
            "z": False,  # forward
            "s": False,  # backward
            "q": False,  # left
            "d": False,  # right
            "a": False,  # up
            "e": False,  # down
            # rotation control keys (same as QWERTY)
            "i": False,
            "k": False,
            "j": False,
            "l": False,
            "u": False,
            "o": False,
            # gripper control keys (AZERTY)
            "w": False,  # close gripper
            "x": False,  # open gripper
        }

    def connect(self):
        if self.connected:
            return

        from pynput import keyboard

        self.listener = keyboard.Listener(
            on_press=self._on_press, on_release=self._on_release
        )

        self.listener_thread = threading.Thread(target=self._start_listener)
        self.listener_thread.daemon = True
        self.listener_thread.start()

        self.connected = True
        print(f"[{self.__class__.__name__}] Connected.")
        print(f"""[{self.__class__.__name__}] Key Bindings (AZERTY):
  - ZQSD : XY movement
  - AE   : Z-axis movement
  - IJKL : Roll and Pitch rotation
  - UO   : Yaw rotation
  - W/X  : Gripper open/close""")

    def set_command_data(self):
        # Arm movement
        delta_pos = np.zeros(3)
        # X-axis (forward/backward)
        if self.state["z"]:
            delta_pos[0] += self.pos_scale
        if self.state["s"]:
            delta_pos[0] -= self.pos_scale
        # Y-axis (left/right)
        if self.state["q"]:
            delta_pos[1] += self.pos_scale
        if self.state["d"]:
            delta_pos[1] -= self.pos_scale
        # Z-axis (up/down)
        if self.state["a"]:
            delta_pos[2] += self.pos_scale
        if self.state["e"]:
            delta_pos[2] -= self.pos_scale

        # Arm rotation (same as parent)
        delta_rpy = np.zeros(3)
        # Roll
        if self.state["j"]:
            delta_rpy[0] -= self.rpy_scale
        if self.state["l"]:
            delta_rpy[0] += self.rpy_scale
        # Pitch
        if self.state["i"]:
            delta_rpy[1] += self.rpy_scale
        if self.state["k"]:
            delta_rpy[1] -= self.rpy_scale
        # Yaw
        if self.state["u"]:
            delta_rpy[2] += self.rpy_scale * 2.0
        if self.state["o"]:
            delta_rpy[2] -= self.rpy_scale * 2.0

        target_se3 = self.arm_manager.target_se3.copy()
        target_se3.translation += delta_pos
        target_se3.rotation = pin.rpy.rpyToMatrix(*delta_rpy) @ target_se3.rotation
        self.arm_manager.set_command_eef_pose(target_se3)

        # Gripper command
        gripper_joint_pos = self.arm_manager.get_command_gripper_joint_pos().copy()
        if self.state["w"] and not self.state["x"]:
            gripper_joint_pos += self.gripper_scale
        elif self.state["x"] and not self.state["w"]:
            gripper_joint_pos -= self.gripper_scale
        self.arm_manager.set_command_gripper_joint_pos(gripper_joint_pos)