"""
Task-specific helpers for RolloutPpoCus.

Each module should expose a `build_ppo_task(rollout, params)` function
that returns an object implementing:

    - `get_extra_state() -> Dict[str, np.ndarray]`
    - optional `on_reset()` hook invoked after environment resets

These helpers allow injecting task-specific state augmentations while
keeping RolloutPpoCus agnostic to the underlying task details.
"""

__all__ = []
