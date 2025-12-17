import logging
from typing import Sequence

import numpy as np
from robo_manip_baselines.common.base.DatasetBase import DatasetBase
from robo_manip_baselines.common.data.RmbData import RmbData

from lerobot.rl.buffer import ReplayBuffer
from lerobot.utils.transition import Transition


class AcfqlDataset(DatasetBase):
    """
    A dummy dataset for ACFQL. This class is primarily a placeholder
    as the main data handling is managed by the LeRobotDataset and
    the custom ReplayBuffer.
    """

    def setup_variables(self):
        """Initializes dataset variables. Not used in this implementation."""
        pass

    def __len__(self):
        """Returns the number of files, which acts as a proxy for dataset size."""
        # This is a placeholder, as the dataset is not loaded into memory here.
        return len(self.filenames)

    def __getitem__(self, idx):
        """Retrieves an item by index. Not used in the main training flow."""
        # Not used in the main training flow.
        return {"file_index": idx}
