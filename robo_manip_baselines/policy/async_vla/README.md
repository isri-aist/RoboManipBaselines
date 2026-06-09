# AsyncVLA

Manipulation port of [AsyncVLA: An Asynchronous VLA for Fast and Robust Navigation on the Edge](https://arxiv.org/abs/2602.13476)
(Hirose et al., [NHirose/AsyncVLA](https://github.com/NHirose/AsyncVLA)).

AsyncVLA decouples a slow, high-level **base VLA** from a fast, reactive onboard
**Edge Adapter**. This implementation adapts that dual-system architecture to
robot-arm manipulation in RoboManipBaselines:

- **Base VLA = pi0** (frozen), loaded via LeRobot exactly like [`policy/pi0`](../pi0).
  It plays the role of the paper's OmniVLA and provides an action chunk as guidance.
- **Edge Adapter = `AsyncVlaPolicy`**, a lightweight policy trained inside
  RoboManipBaselines that refines the (stale) base-VLA guidance using the
  current observation.
- At rollout, pi0 runs in a **background thread**; the Edge Adapter runs every
  control step on the latest (stale) guidance plus a fresh observation.

## Scope and simplifications (v1)

This is the paper's **Stage-1** (base VLA frozen, only the Edge Adapter trained).

- **Guidance = pi0's output action chunk** (`COMMAND_JOINT_POS`, length `N`),
  not the paper's 4096-dim internal embeddings.
- The optical-flow token is replaced by a **delta-image token**
  `concat(I_t, I_{t-k})` (6 channels).
- Base-VLA guidance is **cached offline** so pi0 stays out of the training loop.
- The trajectory re-weighting is an **adaptation** (fixed-base arms have no
  moving local frame): samples are up-weighted when the stale vs. current
  cached guidance diverge beyond a threshold.

Not implemented (future phases): Stage-2 end-to-end / LoRA fine-tuning of pi0,
real-robot remote/edge split, and OpenVLA as the base VLA.

## Prerequisites

- A **trained pi0 checkpoint** for the target task (train it with LeRobot after
  converting the dataset with `misc/ConvertRmbDataToLerobot.py`; see
  [`policy/pi0/README.md`](../pi0/README.md)).
- LeRobot installed in the rollout / caching environment (same setup as pi0).
  The Edge Adapter **training** needs only the base RoboManipBaselines env.

## Usage

Run from `robo_manip_baselines/`. Replace `MujocoUR5eCable` with the env that
matches your pi0 checkpoint's task.

```bash
# 1) Cache the frozen-pi0 guidance into the dataset (pi0 / lerobot env)
uv run python misc/AddPi0GuidanceToRmbData.py <DATASET_DIR> \
    --base_checkpoint <PI0_CKPT> --task_desc "<task>" \
    --camera_names front hand --n_action_steps 8 --guidance_key pi0_guidance

# 2) Train the Edge Adapter (base RMB env, no lerobot)
uv run python ./bin/Train.py AsyncVla --dataset_dir <DATASET_DIR> \
    --state_keys measured_joint_pos --action_keys command_joint_pos \
    --camera_names front hand --skip 1 --n_action_steps 8 \
    --delay_min 1 --delay_max 6 --dth 0.1 --smooth_weight 0.1 \
    --batch_size 64 --num_epochs 200 --lr 1e-4

# 3) Rollout with threaded asynchronous pi0 (pi0 / lerobot env)
uv run python ./bin/Rollout.py AsyncVla MujocoUR5eCable \
    --checkpoint <EDGE_CKPT>/policy_best.ckpt --base_checkpoint <PI0_CKPT> \
    --task_desc "<task>" --world_idx 0 --auto_exit
```

At the end of rollout, `print_statistics` reports the base-VLA inference count,
the control-step count, and the guidance staleness (in steps). True asynchrony
shows up as a base-VLA inference count far below the control-step count, with a
positive, fluctuating staleness whose maximum stays within `--delay_max`.

## Tips

- Set `--delay_max` (training) to cover the staleness you actually observe at
  rollout. If the reported max staleness exceeds `--delay_max`, raise it and
  retrain so the Edge Adapter sees in-distribution delays.
- pi0 and the Edge Adapter share the GPU, so per-step latency jitters while pi0
  runs; this is expected in simulation. Real latency isolation needs separate
  devices (real-robot phase).
