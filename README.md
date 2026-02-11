
# Setup

JAX supports many backends, including [CPU and AMD GPUs](https://docs.jax.dev/en/latest/installation.html).

For this guide, I will list the steps for an NVIDIA GPU (tested on a laptop with an RTX 5070ti mobile GPU).

## Clone

```sh
git clone --recurse-submodules https://github.com/avikde/franka_pipeline_demo.git
```

The Franka model will be at: mujoco_menagerie/franka_emika_panda/

## Setup for NVIDIA GPU acceleration

Inside WSL2: Verify GPU access
```sh
nvidia-smi
```
You should see your NVIDIA GPU listed. If not, update your Windows NVIDIA drivers.

Install Python 3.11+ and dependencies

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3 python3-venv python3-pip build-essential git

cd franka_pipeline_demo
python3 -m venv venv # or place elsewhere
source venv/bin/activate
```

Install MuJoCo + MJX (JAX GPU version)
```bash
# Install JAX with CUDA support (for your NVIDIA GPU)
pip install --upgrade pip
pip install "jax[cuda12]"

# Install MuJoCo (CPU) and MuJoCo-MJX (GPU-accelerated)
pip install mujoco
pip install mujoco-mjx

# Install visualization and utilities
pip install dm_control matplotlib numpy
```

Verify MJX GPU access

```sh
python -c "import jax; print('JAX backend:', jax.default_backend()); print('JAX devices:', jax.devices())"
```

If it doesn't list `CudaDevice` there is an issue.

## Running


