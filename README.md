<p align="center">
  <a href="https://isri-aist.github.io/RoboManipBaselines-ProjectPage">
    <img src="https://github.com/user-attachments/assets/4a4a6429-7e3c-4cfa-90b3-ed54dd9019a3" alt="logo" width="300">
  </a>
  <br/>
  <a href="https://github.com/isri-aist/RoboManipBaselines/actions/workflows/install.yml">
    <img src="https://github.com/isri-aist/RoboManipBaselines/actions/workflows/install.yml/badge.svg" alt="CI-install">
  </a>
  <a href="https://github.com/isri-aist/RoboManipBaselines/actions/workflows/pre-commit.yml">
    <img src="https://github.com/isri-aist/RoboManipBaselines/actions/workflows/pre-commit.yml/badge.svg" alt="CI-pre-commit">
  </a>
  <a href="https://github.com/isri-aist/RoboManipBaselines/blob/master/LICENSE">
    <img src="https://img.shields.io/github/license/isri-aist/RoboManipBaselines" alt="LICENSE">
  </a>
</p>

---

# 🤖 [RoboManipBaselines](https://isri-aist.github.io/RoboManipBaselines-ProjectPage)

<a href="https://isri-aist.github.io/RoboManipBaselines-ProjectPage/"><strong>Project Page</strong></a> |
<a href="https://arxiv.org/abs/2509.17057"><strong>arXiv</strong></a>

A software framework integrating various **imitation learning methods** and **benchmark environments** for robotic manipulation.  
Provides easy-to-use **baselines** for policy training, evaluation, and deployment.

https://github.com/user-attachments/assets/1ad3dcbe-bb4c-4711-96c3-5e196e9d067b

---

## 🚀 Quick Start

Start collecting data in the **MuJoCo** simulation, train your model, and rollout the ACT policy in just a few steps!  
📄 See the [Quick Start Guide](./doc/quick_start.md).

---

## ⚙️ Installation

This project is managed with [uv](https://docs.astral.sh/uv/). Quick start:

```bash
git clone git@github.com:isri-aist/RoboManipBaselines.git --recursive
cd RoboManipBaselines
uv sync                  # base / common install
uv sync --extra act      # add an optional policy (e.g. ACT)
```

For the full per-policy / per-env / per-device setup (including the `third_party/` submodules), follow the step-by-step [Installation Guide](./doc/install.md).

---

## 🧠 Policies

We provide several powerful policy architectures for manipulation tasks:

- 🔹 **[MLP](./robo_manip_baselines/policy/mlp)**: Simple feedforward policy
- 🔹 **[SARNN](./robo_manip_baselines/policy/sarnn)**: Recurrent policy for sequential data
- 🔹 **[ACT](./robo_manip_baselines/policy/act)**: Transformer-based action chunking policy
- 🔹 **[MT-ACT](./robo_manip_baselines/policy/mt_act)**: Multi-task Transformer-based imitation policy
- 🔹 **[Diffusion Policy](./robo_manip_baselines/policy/diffusion_policy)**: Diffusion-based imitation policy
- 🔹 **[3D Diffusion Policy](./robo_manip_baselines/policy/diffusion_policy_3d)**: Diffusion-based policy with 3D point cloud input
- 🔹 **[Flow Policy](./robo_manip_baselines/policy/flow_policy)**: Flow-matching-based policy with 3D point cloud input
- 🔹 **[ManiFlow Policy](./robo_manip_baselines/policy/mani_flow_policy)**: Flow-matching and consistency-based policy with 2D/3D vision
- 🔹 **[pi0](./robo_manip_baselines/policy/pi0)**: Flow-matching-based policy with a vision-language model backbone (PaliGemma)
- 🔹 **[GR00T](./robo_manip_baselines/policy/gr00t)**: Diffusion Transformer-based policy with a vision-language model backbone (Eagle-2)

---

## 📦 Data

- 📂 [Dataset List](./doc/dataset_list.md): Pre-collected expert demonstration datasets
- 🧠 [Learned Parameters](./doc/learned_parameters.md): Trained model checkpoints and configs
- 📄 [Data Format](./doc/rmb_data_format.md): Description of the custom RMB data format used in RoboManipBaselines
- 🪄 [Point Cloud Preprocessing](./doc/preprocessing_pointcloud.md): Data preprocessing for 3D point cloud policies

---

## 🎮 Teleoperation

Use your own teleop interface to collect expert data.  
See [Teleop Tools](./robo_manip_baselines/teleop) for more info.

- 🎮 [Multiple SpaceMouse](./doc/use_multiple_spacemouse.md): Setup multiple SpaceMouse for high-degree-of-freedom robots

---

## 🌍 Environments

Explore diverse manipulation environments:

- 📚 [Environment Catalog](./doc/environment_catalog.md): Overview of all task environments
- 🔧 [Env Setup](./robo_manip_baselines/envs): Installation guides per environment
- ✏️ [How to Add a New Environment](./doc/how_to_add_env.md): Guide for adding a custom environment
- 🔅️ [MuJoCo Tactile Sensor](./doc/mujoco_tactile_sensor.md): Guide for using tactile sensors in MuJoCo environments

---

## 🧰 Miscellaneous

Check out [Misc Scripts](./robo_manip_baselines/misc) for standalone tools and utilities.

---

## 📊 Evaluation Results

See [Benchmarked Performance](./doc/evaluation_results.md) across environments and policies.

---

## 🤝 Contributing

We welcome contributions!  
Check out the [Contribution Guide](./CONTRIBUTING.md) to get started.

---

## 📄 License

This repository is licensed under the **BSD 2-Clause License**, unless otherwise stated.  
Please check individual files or directories (especially `third_party` and `assets`) for specific license terms.

---

## 📖 Citation

If you use RoboManipBaselines in your work, please cite our paper:  
[arXiv](https://arxiv.org/abs/2509.17057) | [Hugging Face](https://huggingface.co/papers/2509.17057):

```bibtex
@article{RoboManipBaselines_Murooka_2025,
  title={RoboManipBaselines: A Unified Framework for Imitation Learning in Robotic Manipulation across Real and Simulation Environments},
  author={Murooka, Masaki and Motoda, Tomohiro and Nakajo, Ryoichi and Oh, Hanbit and Makihara, Koshi and Shirai, Keisuke and Ogata, Tetsuya and Domae, Yukiyasu},
  journal={arXiv preprint arXiv:2509.17057},
  year={2025}
}
```

---
