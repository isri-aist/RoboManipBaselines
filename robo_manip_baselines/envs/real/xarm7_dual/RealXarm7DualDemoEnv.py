import numpy as np

from .RealXarm7DualEnvBase import RealXarm7DualEnvBase


class RealXarm7DualDemoEnv(RealXarm7DualEnvBase):
    def __init__(
        self,
        **kwargs,
    ):
        RealXarm7DualEnvBase.__init__(
            self,
            # Left robot arm index:0, Right robot arm index:1 . This is the same system as UR5eDual.
            init_qpos=np.concatenate(
                [
                    np.deg2rad([0.0, -30.0, 0.0, 45.0, 0.0, 75.0, 0.0]),
                    np.array([800.0]),
                    np.deg2rad([0.0, -30.0, 0.0, 45.0, 0.0, 75.0, 0.0]),
                    np.array([800.0]),
                ]
            ),
            **kwargs,
        )

    def get_input_device_kwargs(self, input_device_name):
        if input_device_name == "spacemouse":
            return {
                0: {"gripper_scale": 20.0},
                1: {"gripper_scale": 20.0},
            }
        else:
            return super().get_input_device_kwargs(input_device_name)

    def modify_world(self, world_idx=None, cumulative_idx=None):
        """Modify simulation world depending on world index."""
        # TODO: Automatically set world index according to task variations
        if world_idx is None:
            world_idx = 0
            # world_idx = cumulative_idx % 2
        return world_idx
