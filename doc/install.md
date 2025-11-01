# Install

## Common installation
Install RoboManipBaselines:
```console
$ git clone git@github.com:isri-aist/RoboManipBaselines.git --recursive
$ cd RoboManipBaselines
$ pip install -e .
```

> [!NOTE]
> If you have problems with pip installation, such as excessive time or module version errors, please add the option `--use-deprecated=legacy-resolver`.

> [!NOTE]
> If you have problems installing the Pinocchio library (`pin` module) from `pip` in certain environments (e.g. Ubuntu 20.04), you can also install it via `apt`. See [here](https://stack-of-tasks.github.io/pinocchio/download.html#Install) for details.

This common installation enables data collection by teleoperation in the MuJoCo environments.

## Installation of each policy
Complete [the common installation](#common-installation) first.

### [MLP](../robo_manip_baselines/policy/mlp)
The MLP policy can be used with only a common installation.

### [SARNN](../robo_manip_baselines/policy/sarnn)
Install dependent libraries including [EIPL](https://github.com/ogata-lab/eipl):
```console
# Go to the top directory of this repository
$ pip install -e .[sarnn]

# Go to the top directory of this repository
$ cd third_party/eipl
$ pip install -e .
```

### [ACT](../robo_manip_baselines/policy/act)
Install dependent libraries including [ACT](https://github.com/tonyzhaozh/act):
```console
# Go to the top directory of this repository
$ pip install -e .[act]

# Go to the top directory of this repository
$ cd third_party/act/detr
$ pip install -e .
```

### [MT-ACT](../robo_manip_baselines/policy/mt_act)
Install dependent libraries including [RoboAgent](https://github.com/robopen/roboagent):
```console
# Go to the top directory of this repository
$ pip install -e .[mt-act]

# Go to the top directory of this repository
$ cd third_party/roboagent/detr
$ pip install -e .
```

### [Diffusion policy](../robo_manip_baselines/policy/diffusion_policy)
Install dependent libraries including [diffusion policy](https://github.com/real-stanford/diffusion_policy):
```console
$ sudo apt install -y libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf

# Go to the top directory of this repository
$ pip install -e .[diffusion-policy]

# Go to the top directory of this repository
$ cd third_party/diffusion_policy
$ pip install -e .
```

> [!NOTE]
> If you encounter the following error,
> ```python
> pip._vendor.packaging.requirements.InvalidRequirement: Expected end or semicolon (after version specifier)
>     opencv-python>=3.
> ```
> replace all `opencv-python>=3.` with `opencv-python>=3.0` in `<venv_directory>/lib/python3.8/site-packages/gym-0.21.0-py3.8.egg-info/requires.txt`.

### [3D diffusion policy](../robo_manip_baselines/policy/diffusion_policy_3d)
Install dependent libraries including [3D diffusion policy](https://github.com/YanjieZe/3D-Diffusion-Policy):
```console
# Go to the top directory of this repository
$ pip install -e .[diffusion-policy-3d]

# Go to the top directory of this repository
$ cd third_party/3D-Diffusion-Policy/3D-Diffusion-Policy
$ pip install -e .
$ cd ../third_party/pytorch3d_simplified
$ pip install -e .
```

> [!NOTE]
> If you encounter a CUDA-related error during the final `pip install -e .` step, please run the command with the following environment variable:
> ```
> $ PYTORCH3D_FORCE_NO_CUDA=1 pip install -e .
> ```

### [Flow policy](../robo_manip_baselines/policy/flow_policy)
Install dependent libraries including [FlowPolicy](https://github.com/zql-kk/FlowPolicy):
```console
# Go to the top directory of this repository
$ pip install -e .[flow-policy]

# Go to the top directory of this repository
$ cd third_party/FlowPolicy/FlowPolicy
$ pip install -e .
$ cd ../third_party/pytorch3d_simplified
$ pip install -e .
```

> [!NOTE]
> If you encounter a CUDA-related error during the final `pip install -e .` step, please run the command with the following environment variable:
> ```
> $ PYTORCH3D_FORCE_NO_CUDA=1 pip install -e .
> ```

### [ManiFlow policy](../robo_manip_baselines/policy/mani_flow_policy)
Install dependent libraries including [ManiFlow Policy](https://github.com/geyan21/ManiFlow_Policy):
```console
# Go to the top directory of this repository
$ pip install -e .[maniflow-policy]

# Go to the top directory of this repository
$ cd third_party/ManiFlow_Policy/ManiFlow
$ pip install -e .
$ cd ../third_party/pytorch3d
$ pip install -e .
$ cd ../r3m
$ pip install -e .
```

> [!NOTE]
> If you encounter a CUDA-related error during the final `pip install -e .` step, please run the command with the following environment variable:
> ```
> $ PYTORCH3D_FORCE_NO_CUDA=1 pip install -e .
> ```

## Installation of each teleoperation interface
Complete [the common installation](#common-installation) first.

### [SpaceMouse](https://3dconnexion.com/us/spacemouse)
[SpaceMouse Wireless](https://3dconnexion.com/us/product/spacemouse-wireless) can be used with only a common installation.

### [GELLO](https://wuphilipp.github.io/gello_site)
Install dependent libraries including [gello_software](https://github.com/wuphilipp/gello_software):
```console
# Go to the top directory of this repository
$ cd third_party/gello_software
$ pip install -e .
$ pip install -e third_party/DynamixelSDK/python
```

## Installation of each environment
Complete [the common installation](#common-installation) first.

### [MuJoCo environments](../robo_manip_baselines/envs/mujoco)
The MuJoCo environment can be used with only a common installation.

### [Isaac environments](../robo_manip_baselines/envs/isaac)
Isaac Gym supports only Python 3.6, 3.7 and 3.8.
In Ubuntu 22.04, use Python 3.8 with [pyenv](https://github.com/pyenv/pyenv).

Download and unpack the Isaac Gym package from [here](https://developer.nvidia.com/isaac-gym).

Install Isaac Gym according to `IsaacGym_Preview_4_Package/isaacgym/doc/install.html`:
```console
$ cd IsaacGym_Preview_4_Package/isaacgym/python
$ pip install -e .
```

Confirm that the sample program can be executed.
```console
$ cd IsaacGym_Preview_4_Package/isaacgym/python/examples
$ python joint_monkey.py
```

Isaac Gym and MuJoCo version 3 are conflicted by a file of the same name, `libsdf.so`, which triggers the following error: `undefined symbol: _ZN32pxrInternal_v0_19__pxrReserved__17 SdfValueTypeNamesE`.
Downgrade MuJoCo:
```console
$ pip install mujoco==2.3.7
```

### [Real UR5e environments](../robo_manip_baselines/envs/real/ur5e)
Install dependent libraries including [gello_software](https://github.com/wuphilipp/gello_software):
```console
# Go to the top directory of this repository
$ pip install -e .[real-ur5e]

# Go to the top directory of this repository
$ cd third_party/gello_software
$ pip install -e .
```

See [here](./real_ur5e.md) for instructions on how to operate real robot.

### [Real xArm7 environments](../robo_manip_baselines/envs/real/xarm7)
Install dependent libraries including [gello_software](https://github.com/wuphilipp/gello_software):
```console
# Go to the top directory of this repository
$ pip install -e .[real-xarm7]

# Go to the top directory of this repository
$ cd third_party/gello_software
$ pip install -e .
```
