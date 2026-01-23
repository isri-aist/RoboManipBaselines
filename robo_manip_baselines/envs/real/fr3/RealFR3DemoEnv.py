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
                    0.02443034,
                    -0.55593735,
                    -0.04863814,
                    -2.25622463,
                    -0.03409599,
                    1.74467456,
                    0.80417317,
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
