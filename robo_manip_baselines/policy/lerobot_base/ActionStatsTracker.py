import logging
import numpy as np

logger = logging.getLogger(__name__)

class ActionStatsTracker:
    """Tracks min/max statistics and logs outliers based on a relative threshold."""

    def __init__(self, action_dim, log_enabled=True, threshold=0.05):
        """
        Initializes the tracker.

        Args:
            action_dim (int): The dimension of the action array to track.
            log_enabled (bool): If False, no logging will occur.
            threshold (float): The relative change required to log a new outlier.
        """
        self.action_dim = action_dim
        self.log_enabled = log_enabled
        self.threshold = threshold
        self.stats = {}

    def _initialize_source(self, source_name):
        """Initializes the tracking for a new data source."""
        self.stats[source_name] = {
            "min": np.full(self.action_dim, np.inf, dtype=np.float64),
            "max": np.full(self.action_dim, -np.inf, dtype=np.float64),
        }

    def update_and_log(self, source_name: str, data_array: np.ndarray):
        """
        Updates statistics for a data source and logs only if new min/max values
        (outliers) exceed the defined threshold.
        """
        if not self.log_enabled or data_array.size == 0 or np.all(np.isnan(data_array)):
            return

        if source_name not in self.stats:
            self._initialize_source(source_name)

        current_min = np.nanmin(data_array, axis=0)
        current_max = np.nanmax(data_array, axis=0)

        stored_min = self.stats[source_name]["min"]
        stored_max = self.stats[source_name]["max"]

        # --- Check for new minimums ---
        new_min_indices = np.where(current_min < stored_min)[0]
        for idx in new_min_indices:
            prev_val = stored_min[idx]
            new_val = current_min[idx]

            # Log if it's the first value or if the change exceeds the threshold.
            is_initial = np.isinf(prev_val)
            # Avoid division by zero if the previous value was 0.
            if prev_val == 0:
                relative_change = np.inf if new_val != 0 else 0
            else:
                relative_change = abs((new_val - prev_val) / prev_val)

            if is_initial or relative_change > self.threshold:
                log_msg = (
                    f"OUTLIER: New MIN for '{source_name}' dim {idx}: {new_val:.4f} (previously {prev_val:.4f})"
                )
                if not is_initial:
                    log_msg += f" - change: {relative_change:.2%}"
                logger.info(log_msg)
            stored_min[idx] = new_val

        # --- Check for new maximums ---
        new_max_indices = np.where(current_max > stored_max)[0]
        for idx in new_max_indices:
            prev_val = stored_max[idx]
            new_val = current_max[idx]

            is_initial = np.isinf(prev_val)
            if prev_val == 0:
                relative_change = np.inf if new_val != 0 else 0
            else:
                relative_change = abs((new_val - prev_val) / prev_val)

            if is_initial or relative_change > self.threshold:
                log_msg = (
                    f"OUTLIER: New MAX for '{source_name}' dim {idx}: {new_val:.4f} (previously {prev_val:.4f})"
                )
                if not is_initial:
                    log_msg += f" - change: {relative_change:.2%}"
                logger.info(log_msg)
            stored_max[idx] = new_val
