import numpy as np

from .RealUR10eEnvBase import RealUR10eEnvBase


class RealUR10eDemoEnv(RealUR10eEnvBase):
    def __init__(
        self,
        **kwargs,
    ):
        RealUR10eEnvBase.__init__(
            self,
                init_qpos=np.array(
                [
                    1.57,
                    -1.57,
                    1.57,
                    -1.57,
                    -1.57,
                    -1.57,
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
