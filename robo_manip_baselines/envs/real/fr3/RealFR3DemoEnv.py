import numpy as np

from .RealFR3EnvBase import RealFR3EnvBase


class RealFR3DemoEnv(RealFR3EnvBase):
    def __init__(
        self,
        **kwargs,
    ):
        RealFR3EnvBase.__init__(
            self,
            init_qpos=np.array(
                [
                    0.0,
                    np.deg2rad(-30),
                    0.0,
                    np.deg2rad(-130),
                    0.0,
                    np.deg2rad(100),
                    np.deg2rad(45),
                    0.05,
                ]
            ),
            **kwargs,
        )

    def modify_world(self, world_idx=None, cumulative_idx=None):
        """Modify simulation world depending on world index."""
        # TODO: Automatically set world index according to task variations
        if world_idx is None:
            world_idx = 0
            # world_idx = cumulative_idx % 2
        return world_idx
