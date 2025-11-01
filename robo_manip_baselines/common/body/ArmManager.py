import dataclasses
from typing import Any, Callable, List, Optional

import numpy as np
import numpy.typing as npt
import pinocchio as pin

from ..data.DataKey import DataKey
from ..utils.MathUtils import (
    get_pose_from_se3,
    get_se3_from_pose,
    get_se3_from_rel_pose,
)
from .BodyManagerBase import BodyConfigBase, BodyManagerBase


class ArmManager(BodyManagerBase):
    """
    Manager for single arm with gripper.

    The manager computes forward and inverse kinematics using Pinocchio, a robot modeling library.
    """

    SUPPORTED_DATA_KEYS = [
        DataKey.MEASURED_JOINT_POS,
        DataKey.COMMAND_JOINT_POS,
        DataKey.MEASURED_JOINT_POS_REL,
        DataKey.COMMAND_JOINT_POS_REL,
        DataKey.MEASURED_JOINT_VEL,
        # DataKey.COMMAND_JOINT_VEL,
        # DataKey.MEASURED_JOINT_TORQUE,
        # DataKey.COMMAND_JOINT_TORQUE,
        DataKey.MEASURED_GRIPPER_JOINT_POS,
        DataKey.COMMAND_GRIPPER_JOINT_POS,
        DataKey.MEASURED_GRIPPER_JOINT_POS_REL,
        DataKey.COMMAND_GRIPPER_JOINT_POS_REL,
        DataKey.MEASURED_EEF_POSE,
        DataKey.COMMAND_EEF_POSE,
        DataKey.MEASURED_EEF_POSE_REL,
        DataKey.COMMAND_EEF_POSE_REL,
        # DataKey.MEASURED_EEF_VEL,
        # DataKey.COMMAND_EEF_VEL,
        DataKey.MEASURED_EEF_WRENCH,
        # DataKey.COMMAND_EEF_WRENCH,
    ]

    def __init__(self, env, body_config):
        super().__init__(env, body_config)

        self.pin_model = pin.buildModelFromUrdf(self.body_config.arm_urdf_path)

        if self.body_config.exclude_joint_names is not None:
            exclude_joint_ids = []
            for exclude_joint_name in self.body_config.exclude_joint_names:
                if self.pin_model.existJointName(exclude_joint_name):
                    exclude_joint_ids.append(
                        self.pin_model.getJointId(exclude_joint_name)
                    )
                else:
                    print(
                        "[{self.__class__.__name__}] Joint does not belong to the model: {exclude_joint_name}"
                    )
            fix_joint_pos = np.zeros(self.pin_model.nq)
            self.pin_model = pin.buildReducedModel(
                self.pin_model, exclude_joint_ids, fix_joint_pos
            )

        if self.body_config.arm_root_pose is not None:
            arm_root_se3 = get_se3_from_pose(self.body_config.arm_root_pose)
            self.pin_model.jointPlacements[1] = arm_root_se3.act(
                self.pin_model.jointPlacements[1]
            )
        self.pin_data = self.pin_model.createData()

        self.reset(init=True)

    def reset(self, init=False):
        self.arm_joint_pos = self.body_config.init_arm_joint_pos.copy()
        self.gripper_joint_pos = self.body_config.init_gripper_joint_pos.copy()

        self.forward_kinematics()

        if init:
            self._original_target_se3 = self.pin_data.oMi[
                self.body_config.ik_eef_joint_id
            ].copy()

        self.target_se3 = self._original_target_se3.copy()

    def set_command_data(self, key, command, is_skip=False):
        if key == DataKey.COMMAND_JOINT_POS:
            self.set_command_joint_pos(
                command[self.body_config.arm_joint_idxes],
                command[self.body_config.gripper_joint_idxes],
            )
        elif key == DataKey.COMMAND_JOINT_POS_REL:
            self.set_command_joint_pos_rel(
                command[self.body_config.arm_joint_idxes],
                command[self.body_config.gripper_joint_idxes],
                is_skip,
            )
        elif key == DataKey.COMMAND_GRIPPER_JOINT_POS:
            gripper_joint_pos = command[
                self.body_config.gripper_joint_idxes_in_gripper_joint_pos
            ]
            self.set_command_gripper_joint_pos(gripper_joint_pos)
        elif key == DataKey.COMMAND_GRIPPER_JOINT_POS_REL:
            gripper_joint_pos_rel = command[
                self.body_config.gripper_joint_idxes_in_gripper_joint_pos
            ]
            self.set_command_gripper_joint_pos_rel(gripper_joint_pos_rel, is_skip)
        elif key == DataKey.COMMAND_EEF_POSE:
            if isinstance(command, pin.SE3):
                eef_pose = command
            elif isinstance(command, (list, tuple)) and isinstance(command[0], pin.SE3):
                eef_pose = command[self.body_config.eef_idx]
            else:
                eef_pose = command[
                    7 * self.body_config.eef_idx : 7 * (self.body_config.eef_idx + 1)
                ]
            self.set_command_eef_pose(eef_pose)
        elif key == DataKey.COMMAND_EEF_POSE_REL:
            eef_pose_rel = command[
                6 * self.body_config.eef_idx : 6 * (self.body_config.eef_idx + 1)
            ]
            self.set_command_eef_pose_rel(eef_pose_rel, is_skip)
        else:
            raise ValueError(
                f"[{self.__class__.__name__}] Invalid command data key: {key}"
            )

    def set_command_joint_pos(self, arm_joint_pos, gripper_joint_pos):
        self.arm_joint_pos = arm_joint_pos
        self.forward_kinematics()
        self.target_se3 = self.current_se3.copy()
        self.set_command_gripper_joint_pos(gripper_joint_pos)

    def set_command_joint_pos_rel(
        self, arm_joint_pos_rel, gripper_joint_pos_rel, is_skip=False
    ):
        arm_joint_pos = self.arm_joint_pos.copy()
        gripper_joint_pos = self.gripper_joint_pos.copy()
        if not is_skip:
            arm_joint_pos += arm_joint_pos_rel
            gripper_joint_pos += gripper_joint_pos_rel
        self.set_command_joint_pos(arm_joint_pos, gripper_joint_pos)

    def set_command_gripper_joint_pos(self, gripper_joint_pos):
        self.gripper_joint_pos = np.clip(
            gripper_joint_pos,
            self.env.action_space.low[self.body_config.gripper_joint_idxes_for_limit],
            self.env.action_space.high[self.body_config.gripper_joint_idxes_for_limit],
        )

    def set_command_gripper_joint_pos_rel(self, gripper_joint_pos_rel, is_skip=False):
        gripper_joint_pos = self.gripper_joint_pos.copy()
        if not is_skip:
            gripper_joint_pos += gripper_joint_pos_rel
        self.set_command_gripper_joint_pos(gripper_joint_pos)

    def set_command_eef_pose(self, eef_pose):
        if isinstance(eef_pose, pin.SE3):
            self.target_se3 = eef_pose
        else:
            self.target_se3 = get_se3_from_pose(eef_pose)
        self.inverse_kinematics()

    def set_command_eef_pose_rel(self, eef_pose_rel, is_skip=False):
        target_se3 = self.target_se3.copy()
        if not is_skip:
            target_se3 = target_se3 * get_se3_from_rel_pose(eef_pose_rel)
        self.set_command_eef_pose(target_se3)

    def get_eef_pose_from_joint_pos(self, arm_joint_pos):
        pin_data = self.pin_model.createData()
        pin.forwardKinematics(self.pin_model, pin_data, arm_joint_pos)
        se3 = pin_data.oMi[self.body_config.ik_eef_joint_id]
        return get_pose_from_se3(se3)

    def get_command_data(self, key):
        if key == DataKey.COMMAND_JOINT_POS:
            return self.get_command_joint_pos()
        elif key == DataKey.COMMAND_GRIPPER_JOINT_POS:
            return self.get_command_gripper_joint_pos()
        elif key == DataKey.COMMAND_EEF_POSE:
            return self.get_command_eef_pose()
        else:
            raise ValueError(
                f"[{self.__class__.__name__}] Invalid command data key: {key}"
            )

    def get_command_joint_pos(self):
        return (self.arm_joint_pos, self.gripper_joint_pos)

    def get_command_gripper_joint_pos(self):
        return self.gripper_joint_pos

    def get_command_eef_pose(self):
        return get_pose_from_se3(self.target_se3)

    def draw_markers(self):
        if self.body_config.get_root_pose_func is None:
            target_se3 = self.target_se3
            current_se3 = self.current_se3
        else:
            root_pose_rel = (
                self.body_config.get_root_pose_func(self.env.unwrapped)
                * get_se3_from_pose(self.body_config.arm_root_pose).inverse()
            )
            target_se3 = root_pose_rel * self.target_se3
            current_se3 = root_pose_rel * self.current_se3

        self.env.unwrapped.draw_box_marker(
            pos=target_se3.translation,
            mat=target_se3.rotation,
            size=(0.02, 0.02, 0.03),
            rgba=(0, 1, 0, 0.5),
        )
        self.env.unwrapped.draw_box_marker(
            pos=current_se3.translation,
            mat=current_se3.rotation,
            size=(0.02, 0.02, 0.03),
            rgba=(1, 0, 0, 0.5),
        )

    def forward_kinematics(self):
        pin.forwardKinematics(
            self.pin_model,
            self.pin_data,
            self.arm_joint_pos,
        )

    def inverse_kinematics(self):
        # https://gepettoweb.laas.fr/doc/stack-of-tasks/pinocchio/master/doxygen-html/md_doc_b-examples_d-inverse-kinematics.html
        error_se3 = self.current_se3.actInv(self.target_se3)
        error_vec = pin.log(error_se3).vector  # in joint frame
        arm_joint_pos = self.arm_joint_pos
        J = pin.computeJointJacobian(
            self.pin_model,
            self.pin_data,
            arm_joint_pos,
            self.body_config.ik_eef_joint_id,
        )  # in joint frame
        J = -1 * np.dot(pin.Jlog6(error_se3.inverse()), J)
        damping_scale = 1e-6
        delta_arm_joint_pos = -1 * J.T.dot(
            np.linalg.solve(
                J.dot(J.T)
                + (np.dot(error_vec, error_vec) + damping_scale) * np.identity(6),
                error_vec,
            )
        )
        self.arm_joint_pos = pin.integrate(
            self.pin_model, arm_joint_pos, delta_arm_joint_pos
        )
        self.forward_kinematics()

    @property
    def current_se3(self):
        return self.pin_data.oMi[self.body_config.ik_eef_joint_id]


@dataclasses.dataclass
class ArmConfig(BodyConfigBase):
    """Configuration for single arm with gripper."""

    BodyManagerClass = ArmManager

    # URDF file loaded for robot model in Pinocchio library
    arm_urdf_path: str

    # Arm root pose
    arm_root_pose: Optional[npt.NDArray[np.float64]]

    # Link ID of end-effector in robot model when solving IK by Pinocchio library
    ik_eef_joint_id: int

    # Indices of arm joints at observed joint positions
    arm_joint_idxes: npt.NDArray[np.int_]

    # Indices of gripper joints at observed joint positions
    gripper_joint_idxes: npt.NDArray[np.int_]

    # Indicies of gripper joints (belonging to this body config) in a vector of gripper joint positions only
    gripper_joint_idxes_in_gripper_joint_pos: npt.NDArray[np.int_]

    # Index of end-effector in observed end-effector poses
    eef_idx: Optional[int]

    # Initial arm joint positions
    init_arm_joint_pos: npt.NDArray[np.float64]

    # Initial gripper joint positions
    init_gripper_joint_pos: npt.NDArray[np.float64]

    # [Optional] Indices of gripper joints at joint positions limits (high/low)
    gripper_joint_idxes_for_limit: Optional[npt.NDArray[np.int_]] = None

    # [Optional] Joints to be excluded from the URDF model when building robot model for Pinocchio library
    exclude_joint_names: Optional[List[str]] = None

    # [Optional] Function to get the current arm root pose (used only for drawing markers)
    get_root_pose_func: Optional[Callable[..., Any]] = None

    def __post_init__(self):
        if self.gripper_joint_idxes_for_limit is None:
            self.gripper_joint_idxes_for_limit = self.gripper_joint_idxes
