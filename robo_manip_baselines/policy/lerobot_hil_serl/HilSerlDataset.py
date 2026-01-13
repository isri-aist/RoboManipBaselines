import logging
from typing import Sequence

import numpy as np
from robo_manip_baselines.common.base.DatasetBase import DatasetBase
from robo_manip_baselines.common.data.RmbData import RmbData

from lerobot.rl.buffer import ReplayBuffer
from lerobot.utils.transition import Transition


class HilSerlDataset(DatasetBase):
    """
    A dummy dataset for HIL-SERL.
    """

    def setup_variables(self):
        pass

    def __len__(self):
        # This is a placeholder, as the dataset is not loaded into memory here.
        return len(self.filenames)

    def __getitem__(self, idx):
        # Not used in the main training flow.
        return {"file_index": idx}
