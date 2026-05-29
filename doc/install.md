# Install

This project is managed with [uv](https://docs.astral.sh/uv/). Install it first if you do not have it:

```console
$ curl -LsSf https://astral.sh/uv/install.sh | sh
```

All commands below assume the project virtual environment is active (`source .venv/bin/activate`) or that each command is prefixed with `uv run`.

> [!IMPORTANT]
> Clone with `--recursive` (or run `git submodule update --init --recursive` after cloning) before running `uv sync`.
> All third-party packages under `third_party/` are declared as editable path sources in `pyproject.toml` ([tool.uv.sources](../pyproject.toml)), so `uv sync --extra <name>` installs them automatically — no `uv pip install` step is required.

## Common installation
Install RoboManipBaselines:
```console
$ git clone git@github.com:isri-aist/RoboManipBaselines.git --recursive
$ cd RoboManipBaselines
$ uv sync
```

> [!NOTE]
> If you have problems installing the Pinocchio library (`pin` module) in certain environments (e.g. Ubuntu 20.04), you can also install it via `apt`. See [here](https://stack-of-tasks.github.io/pinocchio/download.html#Install) for details.

This common installation enables data collection by teleoperation in the MuJoCo environments.

## Installation of each policy
Complete [the common installation](#common-installation) first.

### [MLP](../robo_manip_baselines/policy/mlp)
The MLP policy can be used with only a common installation.

### [SARNN](../robo_manip_baselines/policy/sarnn)
Includes [EIPL](https://github.com/ogata-lab/eipl):
```console
$ uv sync --extra sarnn
```

### [ACT](../robo_manip_baselines/policy/act)
Includes [ACT](https://github.com/tonyzhaozh/act):
```console
$ uv sync --extra act
```

### [MT-ACT](../robo_manip_baselines/policy/mt_act)
Includes [RoboAgent](https://github.com/robopen/roboagent):
```console
$ uv sync --extra mt-act
```

> [!NOTE]
> The `act` and `mt-act` extras both ship a package named `detr` from different upstreams, so they are declared mutually exclusive in `pyproject.toml` (`[tool.uv] conflicts`). Use one at a time, e.g. `uv sync --extra act` or `uv sync --extra mt-act`, not both.

### [Diffusion policy](../robo_manip_baselines/policy/diffusion_policy)
Includes [diffusion policy](https://github.com/real-stanford/diffusion_policy):
```console
$ sudo apt install -y libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf
$ uv sync --extra diffusion-policy
```

> [!NOTE]
> If you encounter the following error,
> ```python
> pip._vendor.packaging.requirements.InvalidRequirement: Expected end or semicolon (after version specifier)
>     opencv-python>=3.
> ```
> replace all `opencv-python>=3.` with `opencv-python>=3.0` in `<venv_directory>/lib/python3.8/site-packages/gym-0.21.0-py3.8.egg-info/requires.txt`.

### [3D diffusion policy](../robo_manip_baselines/policy/diffusion_policy_3d)
Includes [3D diffusion policy](https://github.com/YanjieZe/3D-Diffusion-Policy):
```console
$ uv sync --extra diffusion-policy-3d
```

> [!NOTE]
> If `pytorch3d` fails to build with a CUDA error, set the env var before syncing:
> ```console
> $ PYTORCH3D_FORCE_NO_CUDA=1 uv sync --extra diffusion-policy-3d
> ```

### [Flow policy](../robo_manip_baselines/policy/flow_policy)
Includes [FlowPolicy](https://github.com/zql-kk/FlowPolicy):
```console
$ uv sync --extra flow-policy
```

> [!NOTE]
> If `pytorch3d` fails to build with a CUDA error, set the env var before syncing:
> ```console
> $ PYTORCH3D_FORCE_NO_CUDA=1 uv sync --extra flow-policy
> ```

### [ManiFlow policy](../robo_manip_baselines/policy/mani_flow_policy)
Includes [ManiFlow Policy](https://github.com/geyan21/ManiFlow_Policy):
```console
$ uv sync --extra maniflow-policy
```

> [!NOTE]
> If `pytorch3d` fails to build with a CUDA error, set the env var before syncing:
> ```console
> $ PYTORCH3D_FORCE_NO_CUDA=1 uv sync --extra maniflow-policy
> ```

> [!NOTE]
> The `diffusion-policy-3d` / `flow-policy` / `maniflow-policy` extras each ship a different `pytorch3d`, so they are declared mutually exclusive in `pyproject.toml` (`[tool.uv] conflicts`). Use only one at a time.

## Installation of each teleoperation device
Complete [the common installation](#common-installation) first.

### [SpaceMouse](https://3dconnexion.com/us/spacemouse)
[SpaceMouse Wireless](https://3dconnexion.com/us/product/spacemouse-wireless) can be used with only a common installation.

### [GELLO](https://wuphilipp.github.io/gello_site)
Includes [gello_software](https://github.com/wuphilipp/gello_software) and the Dynamixel SDK:
```console
$ uv sync --extra gello
```

## Installation of each sensor device
Complete [the common installation](#common-installation) first.

### [Femto Bolt](https://www.orbbec.com/products/tof-camera/femto-bolt/)
[pyorbbecsdk](https://github.com/orbbec/pyorbbecsdk) builds a native library and is not installed via `uv sync`. Install its Python requirements into the project venv and build the C++ side manually:
```console
$ uv add --requirements third_party/pyorbbecsdk/requirements.txt
$ cd third_party/pyorbbecsdk
$ mkdir build
$ cd build
$ cmake -Dpybind11_DIR=`pybind11-config --cmakedir` ..
$ make -j4
$ make install

# Go to the top directory of this repository
$ export PYTHONPATH=$PYTHONPATH:`realpath third_party/pyorbbecsdk/install/lib/`
$ cd third_party/pyorbbecsdk
$ sudo bash ./scripts/install_udev_rules.sh
$ sudo udevadm control --reload-rules && sudo udevadm trigger
```

> [!NOTE]
> Run the following command to set `PYTHONPATH` each time you open a terminal.
> ```console
> # Go to the top directory of this repository
> $ export PYTHONPATH=$PYTHONPATH:`realpath third_party/pyorbbecsdk/install/lib/`
> ```

## Installation of each environment
Complete [the common installation](#common-installation) first.

### [MuJoCo environments](../robo_manip_baselines/envs/mujoco)
The MuJoCo environment can be used with only a common installation.

### [Isaac environments](../robo_manip_baselines/envs/isaac)
Isaac Gym supports only Python 3.6, 3.7 and 3.8. In Ubuntu 22.04, create the venv with Python 3.8:
```console
$ uv venv --python 3.8
$ uv sync
```

Isaac Gym ships as a stand-alone tarball outside the repo, so it is installed into the project venv with `uv add`:
```console
$ uv add --editable IsaacGym_Preview_4_Package/isaacgym/python
```

Confirm that the sample program can be executed.
```console
$ cd IsaacGym_Preview_4_Package/isaacgym/python/examples
$ uv run python joint_monkey.py
```

Isaac Gym and MuJoCo version 3 conflict on a file named `libsdf.so`, which triggers `undefined symbol: _ZN32pxrInternal_v0_19__pxrReserved__17 SdfValueTypeNamesE`. Downgrade MuJoCo by overriding the pin temporarily:
```console
$ uv add 'mujoco==2.3.7'
```

### [TACTO environments](../robo_manip_baselines/envs/tacto)
Includes [tacto](https://github.com/facebookresearch/tacto) and its nested submodules (`attrdict`, `urdfpy`, `pybulletX`):
```console
$ uv sync --extra tacto
```

### [Real UR5e environments](../robo_manip_baselines/envs/real/ur5e)
Includes [gello_software](https://github.com/wuphilipp/gello_software) and the UR RTDE / RealSense / hid stack:
```console
$ uv sync --extra real-ur5e
```

See [here](./real_ur5e.md) for instructions on how to operate real robot.

### [Real xArm7 environments](../robo_manip_baselines/envs/real/xarm7)
Includes [gello_software](https://github.com/wuphilipp/gello_software) and the xArm SDK / RealSense / hid stack:
```console
$ uv sync --extra real-xarm7
```
