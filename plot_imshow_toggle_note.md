#これは未来のTo doというか、対策みたいなファイルです。

# Optional cv2.imshow Toggle for Rollout Plot

## Context
- Real-world rollouts (`RolloutBase` subclasses such as `RolloutPpo` and `RolloutMlp`) always render their matplotlib canvas through `cv2.imshow` each time `draw_plot` runs.
- Even with `--skip_draw` these calls can dominate the per-step latency because they block on GUI refresh. For responsive control (target `dt = 0.02 s`) we may want the plot logic to stay enabled but throttle or disable the actual window blits.
- Directly disabling the plot via `--no_plot` removes the GUI overhead but also prevents key events (`cv2.waitKey`) from triggering the end-of-rollout transition, which breaks `.rmb/.hdf5` saving when we exit with Ctrl+C.

## Goal
Provide a lightweight switch that lets operators keep the plotting pipeline (and `cv2.waitKey`) active while skipping the expensive `cv2.imshow` refresh most of the time. This should:
- Maintain compatibility with the current CLI
- Preserve rollouts saving flow (`EndRolloutPhase` still runs)
- Require minimal changes in core files so the behaviour is easy to reason about

## Proposed Interface
Add a new optional CLI flag on `RolloutBase` (available to all rollouts):

```
--imshow-interval N
```

- `N >= 1`. When `N == 1` (default) behaviour matches the current implementation (refresh every draw call).
- Larger `N` values refresh the OpenCV window only once every `N` calls to `draw_plot`. `--imshow-interval 0` should coerce to `1` to keep the interface safe.

Setting `N` to a large value approximates “imshow off” while keeping the window alive and the keyboard handler working.

## Implementation Sketch
1. **Argument parsing** (`robo_manip_baselines/common/base/RolloutBase.py`)
   - Register `--imshow_interval` next to other plot-related options.
   - Bound-check after parsing (`<= 0` → reset to `1`, print a warning).
2. **State bookkeeping** (also in `RolloutBase`)
   - Store `_imshow_interval` and `_last_imshow_step` in `setup_args` or `setup_variables`.
   - Implement helper methods:
     - `_should_update_imshow(self)`: returns `True` when `rollout_time_idx - _last_imshow_step >= _imshow_interval`.
     - `_show_plot_image(self, image, force=False)`: calls `cv2.imshow` (and `cv2.moveWindow`) only when `force` is `True` or `_should_update_imshow()`.
3. **Replace direct calls**
   - In `RolloutBase.setup_plot`, `RolloutBase.reset`, and each policy’s `draw_plot` (e.g., `policy/ppo/RolloutPpo.py`, `policy/mlp/RolloutMlp.py`), swap direct `cv2.imshow(...)` calls for `_show_plot_image(...)`.
   - Keep `cv2.waitKey(1)` exactly where it is so keyboard interaction and timers are unaffected.
4. **Force initial draw**
   - When the window is first created (or during reset), call `_show_plot_image(..., force=True)` to ensure the window appears even if `_imshow_interval > 1`.

## Expected Behaviour
- Default run: identical to current master (no behaviour change).
- With `--imshow-interval 10`: window updates every 10 draw calls; interactive key handling remains intact; `.hdf5` saving works.
- With `--imshow-interval 1e9` (or similarly large number): window effectively stays frozen while the underlying loop runs at “no_plot” speed; operator can still press `n`/`Esc`.

## Testing Notes
- Unit tests are not present for rollout loops; rely on manual testing.
- Recommended manual checks:
  1. `python ./bin/Rollout.py ...` with default arguments → behaviour unchanged.
  2. Repeat with `--imshow-interval 10` → confirm window redraw cadence slows, keyboard still works, rollout saves correctly.
  3. Use `--imshow-interval 1000000000` → check control loop timing improves without breaking exit/save flow.
- Optionally re-run `--ppo-profile` to record `env_step` vs `draw_plot` impact.

## Open Questions / Follow-ups
- Should the interval apply independently to multiple windows (multi-camera setups)? Current plan uses global rollout step; acceptable for now but could be per-axis if needed.
- If additional throttling is required, consider skipping `self.canvas.draw()` when `cv2.imshow` is skipped, or caching the rendered image until the next redraw.
- For full “no GUI” support while retaining save, revisit the idea of calling `save_data` from a `finally` block in `RolloutBase.run`.
