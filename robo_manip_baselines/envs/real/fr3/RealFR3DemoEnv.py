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
                    1.18000162,
                    -1.91696992,
                    1.5561803,
                    -1.21203147,
                    -1.57465679,
                    -0.39695961,
                    0.0,
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
