# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository overview

RoboManipBaselines is a unified imitation-learning framework for robotic manipulation. It bundles a set of policy architectures (MLP, SARNN, ACT, MT-ACT, Diffusion Policy, 3D Diffusion Policy, Flow Policy, ManiFlow Policy, pi0, GR00T) with simulated environments (MuJoCo, Isaac Gym, TACTO) and real-robot drivers (UR5e, xArm7), all sharing a common data format and training/rollout entry points.

The end-to-end workflow is always: **Teleop → dataset → Train → checkpoint → Rollout**.

## Common commands

This project is managed with **uv**. Either activate the venv (`source .venv/bin/activate`) once per shell, or prefix each command with `uv run` (e.g., `uv run python ./bin/Teleop.py ...`). The examples below assume the venv is already active.

All commands are run from `robo_manip_baselines/` (one level below the repo root) unless stated otherwise.

```bash
# Teleoperation (data collection)
uv run python ./bin/Teleop.py <EnvName> [--input_device {spacemouse,keyboard,gello,vive}] [--world_idx_list 0 5] [--file_format {rmb,hdf5}]

# Training
uv run python ./bin/Train.py <PolicyName> --dataset_dir ./dataset/<dataset_name> [--checkpoint_dir ./checkpoint/...]

# Policy rollout (inference)
uv run python ./bin/Rollout.py <PolicyName> <EnvName> --checkpoint ./checkpoint/<Policy>/<run>/policy_last.ckpt [--world_idx N] [--auto_exit] [--save_rollout]
```

`<PolicyName>` is one of `Mlp Sarnn Act MtAct DiffusionPolicy DiffusionPolicy3d FlowPolicy ManiFlowPolicy Gr00t Pi0` (Train.py supports a subset — see `RolloutMain.policy_choices` and `TrainMain.policy_choices`).
`<EnvName>` is auto-discovered from `robo_manip_baselines/envs/operation/Operation<EnvName>.py` (e.g., `MujocoUR5eCable`, `RealUR5eDemo`, `IsaacUR5eCabinet`, `TactoSawyerGrasp`). Pass `-h` after the subcommand to see policy/env-specific arguments handled by the inner parser.

### Linting / formatting

Pre-commit is the source of truth (no separate `lint` make target):

```bash
pre-commit install                  # one-time setup
pre-commit run --all-files          # run all hooks (ruff --fix --extend-select I, ruff-format, EOL/whitespace/YAML checks)
```

`pre-commit` excludes everything under `third_party/`.

### Tests

There is no `pytest` suite; `robo_manip_baselines/tests/` contains standalone scripts that take a teleop data file as input:

```bash
uv run python ./tests/TestDataUtils.py <path-to-*.rmb-or-.hdf5>
uv run python ./tests/TestRealEnvBaseGetInfo.py
```

CI (`.github/workflows/install.yml`) only verifies that the package installs across Python 3.10–3.12 with each optional-deps matrix entry (`[sarnn]`, `[act]`, `[diffusion-policy]`, `[real-ur5e]`, `[real-xarm7]`).

### Installation pattern (important when adding a policy)

Use **uv** for environment management — `uv sync` reads `pyproject.toml`, creates `.venv/`, and writes/refreshes `uv.lock`. Every third-party package under `third_party/` is declared as an editable path source in `[tool.uv.sources]`, so a single `uv sync --extra <name>` installs both the project extra and its third-party deps. **Do not use `uv pip install`** to add third-party submodules — declare them as path sources instead so they stay in `uv.lock`.

```bash
uv sync                                       # base / common install only
uv sync --extra act                           # base + ACT (incl. third_party/act/detr)
uv sync --extra real-ur5e                     # real UR5e + gello + dynamixel_sdk
uv sync --extra tacto                         # tacto + attrdict + urdfpy + pybulletX
```

The available extras mirror `[project.optional-dependencies]` in `pyproject.toml`: `sarnn`, `act`, `mt-act`, `diffusion-policy`, `diffusion-policy-3d`, `flow-policy`, `maniflow-policy`, `tacto`, `gello`, `real-ur5e`, `real-xarm7`, `benchmarks`.

**Conflicting extras** (declared via `[tool.uv] conflicts`): `act` ⟷ `mt-act` (both ship a `detr` package from different upstreams), and `diffusion-policy-3d` ⟷ `flow-policy` ⟷ `maniflow-policy` (each ships a different `pytorch3d`). uv resolves a separate environment per combination — never use both sides of a conflict pair at once.

**Build-time dependencies**: several third-party packages (`pytorch3d`, `r3m`, `detr`, `eipl`, the three 3D-policy packages) import `torch` from their setup.py. Build isolation hides torch by default, so `[tool.uv.extra-build-dependencies]` injects `torch` into their build environment. Add new entries there if a freshly-added third-party package fails to build with `ModuleNotFoundError: No module named 'torch'` (or similar).

`doc/install.md` and `doc/quick_start.md` document the same workflow per-policy / per-env / per-device using only `uv sync --extra X`. The only cases that still need imperative install commands are Isaac Gym (an out-of-tree tarball from NVIDIA, added with `uv add --editable`) and Femto Bolt (a native C++ build). CI (`.github/workflows/install.yml`) still uses `pip install` directly — keep that as-is unless deliberately switching CI to uv.

Always clone with `--recursive` or run `git submodule update --init --recursive` first; `.gitmodules` lists ~12 submodules under `third_party/` and several of them carry their own nested submodules that uv expects to be present at sync time.

## Architecture

### Meta-parser → dynamic dispatch → multiple inheritance

`bin/Teleop.py`, `bin/Train.py`, and `bin/Rollout.py` are deliberately thin. Each:

1. Uses an outer `argparse` that *only* parses the policy and/or env name, then rewrites `sys.argv` and forwards the rest to an inner parser owned by the dispatched class.
2. Dynamically imports the right module via `importlib`: `Train.py` maps `Act` → `robo_manip_baselines.policy.act` using `camel_to_snake` (`common/utils/MiscUtils.py`); `Rollout.py`/`Teleop.py` resolve `Operation<EnvName>` from `robo_manip_baselines.envs.operation`.
3. Composes a runtime class by multiple inheritance, e.g.:

   ```python
   class Rollout(OperationEnvClass, RolloutPolicyClass): ...
   class Teleop(OperationEnvClass, TeleopBase): ...
   ```

   **The MRO order matters** and is documented as such in the source — the operation class is always first so that its `setup_env` and `get_pre_motion_phases` win, while the policy/teleop class supplies `infer_policy`/`run`. Preserve this order when extending.

When `<EnvName>` ends with `Vec` or starts with `Isaac`, `Teleop.py` swaps in `TeleopBaseVec` and imports `isaacgym` before any PyTorch import (Isaac Gym must initialize before torch). Treat this ordering as load-bearing.

### Core abstractions (`robo_manip_baselines/common/`)

- **`base/TrainBase.py`** — abstract trainer. Owns argparse defaults (state/action keys, augmentation stds, normalization type, skip, batch size, epochs, lr). Subclasses set `DatasetClass` and implement `setup_policy()` + `train_loop()`. `set_additional_args(parser)` is the override point for policy-specific CLI flags and `parser.set_defaults(...)`.
- **`base/RolloutBase.py`** — abstract rollout runner. Phase pipeline is fixed: `InitialRolloutPhase → get_pre_motion_phases() (from the env operation class) → RolloutPhase → EndRolloutPhase`. Subclasses must implement `setup_policy()`, `infer_policy()`, and `draw_plot()`.
- **`base/PhaseBase.py`** — phases drive the control loop. Override `start`, `pre_update`, `post_update`, `check_transition`. `ReachPhaseBase` and `GraspPhaseBase` are the standard pre-motion phases used by Operation classes (e.g., `OperationMujocoUR5eCable` defines `ReachPhase1/ReachPhase2/GraspPhase`).
- **`data/DataKey.py`** — canonical string keys for everything in the RMB file (`MEASURED_JOINT_POS`, `COMMAND_EEF_POSE`, `MEASURED_EEF_WRENCH`, …) and a single source of truth for dimensions (`get_dim`, `get_dim_for_policy`, where EEF pose is stored as 7-vector but presented to policies as 9-vector), relative/absolute key conversion, and image key naming. Always look here before inventing a new field.
- **`data/RmbData.py`** — context-manager loader that abstracts over the two on-disk variants (`.rmb` directory of mp4+hdf5 vs `.hdf5` single file). Has optional in-process video caching.
- **`manager/`** — `DataManager` records and persists episodes; `MotionManager` translates between command/measured representations and the env's action space; `PhaseManager` runs the phase state machine.
- **`body/`** — `BodyConfig`/`BodyManager` describe per-body kinematics; `ArmConfig` declares which joint indices belong to the arm vs gripper and which EEF index is active. Multi-arm and mobile-base envs compose multiple `ArmConfig`/`MobileOmniConfig` entries in `env.unwrapped.body_config_list`.

### Adding components

- **New policy**: create `robo_manip_baselines/policy/<snake_name>/` with `Train<Name>.py`, `Rollout<Name>.py`, `<Name>Dataset.py`, `<Name>Policy.py` (optional), and an `__init__.py` re-exporting them. Add `<Name>` to `TrainMain.policy_choices` and/or `RolloutMain.policy_choices` in `bin/Train.py`/`bin/Rollout.py`. Use an existing simple policy (`policy/mlp/`) as the template.
- **New environment**: follow `doc/how_to_add_env.md` — env class under `envs/mujoco/<robot>/`, `Operation<EnvName>` under `envs/operation/`, MuJoCo XML under `envs/assets/mujoco/envs/...`, then re-export from `envs/mujoco/__init__.py` and register the gym id in `envs/__init__.py`.

### Checkpoints and model meta info

Trained policies persist `model_meta_info.pkl` alongside `policy_*.ckpt` in `robo_manip_baselines/checkpoint/<Policy>/<run>/`. `RolloutBase.setup_model_meta_info()` reads it to recover `state_keys`, `action_keys`, `camera_names`, dataset stats, augmentation config, and per-policy args. Treat this metadata blob as part of the checkpoint contract — never load a checkpoint without it.

### RMB data format

Two on-disk variants share the same logical schema (see `doc/rmb_data_format.md`):

- **RmbData-Compact** (`*.rmb/`): a directory holding `main.rmb.hdf5` plus one `<camera>_rgb_image.rmb.mp4` and `<camera>_depth_image.rmb.mp4` per camera. Depth is quantized to 1 mm via `videoio`.
- **RmbData-SingleHDF5** (`*.hdf5`): one file with everything.

Conversion and visualization scripts live in `robo_manip_baselines/misc/` (`SwitchRmbDataFormat.py`, `CompareRmbData.py`, `VisualizeData.py`, `AddPointCloudToRmbData.py`, `ConvertRmbDataToLerobot.py`, …).

## Conventions

- Python ≥ 3.10 is the floor (raised from 3.8 when `torch==2.10.0` was pinned — that wheel requires 3.10+). CI tests 3.10–3.12. `MiscUtils.remove_prefix`/`remove_suffix` are leftover 3.8 polyfills for `str.removeprefix`/`removesuffix`; they still work and have callers, so do not delete them blindly.
- Class-name ↔ module-name mapping is `CamelCase` → `snake_case` via `camel_to_snake` (handles cases like `DiffusionPolicy3d` → `diffusion_policy_3d`). Keep new policy directories consistent with this rule or `Train.py`/`Rollout.py` dispatch will break.
- The framework relies on `pinocchio` (`pin`) for IK; if the wheel install of `pin` fails on older Ubuntu, install it via `apt` as noted in `doc/install.md`.
- Isaac Gym envs require Python 3.8 and conflict with MuJoCo 3 (`libsdf.so` clash) — downgrade to `mujoco==2.3.7` when using them.
