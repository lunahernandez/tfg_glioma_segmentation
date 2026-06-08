# SegMamba setup notes

This document describes the additional setup required to run the SegMamba-based experiments in this project.

SegMamba was installed locally from the original repository. Due to compatibility issues with the NVIDIA RTX 5090, `causal-conv1d` and `mamba-ssm` had to be compiled manually with support for compute capability `sm_120`.

## Environment used

* OS: WSL Ubuntu
* GPU: NVIDIA RTX 5090
* CUDA Toolkit: 12.8
* PyTorch: 2.11.0+cu128
* MONAI: 1.5.2
* transformers: 4.36.2

## Important note

`causal-conv1d` and `mamba-ssm` are not included in `requirements.txt` because they must be installed locally from the SegMamba repository after applying the compatibility changes described below.

In the commands below, replace `/path/to/tfg_glioma_segmentation` with the local path where this repository was cloned.

```bash
export PROJECT_ROOT=/path/to/tfg_glioma_segmentation
```

## 1. Install CUDA Toolkit 12.8 under WSL

SegMamba requires local compilation of CUDA-dependent packages. For this reason, CUDA Toolkit 12.8 was installed under WSL.

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600

sudo mkdir -p /usr/share/keyrings
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu-keyring.gpg
sudo mv cuda-wsl-ubuntu-keyring.gpg /usr/share/keyrings/

echo "deb [signed-by=/usr/share/keyrings/cuda-wsl-ubuntu-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/ /" | sudo tee /etc/apt/sources.list.d/cuda-wsl-ubuntu-x86_64.list

sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-8
```

These commands:

* add the NVIDIA CUDA package repository for WSL;
* register the repository signing key;
* update the local package index;
* install CUDA Toolkit 12.8, including `nvcc`, CUDA libraries and header files.

## 2. Configure CUDA 12.8 as the active CUDA version

```bash
sudo ln -sfn /usr/local/cuda-12.8 /usr/local/cuda

echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc

source ~/.bashrc
hash -r
nvcc -V
```

These commands:

* create `/usr/local/cuda` as a symbolic link to CUDA 12.8;
* define `CUDA_HOME`, which is used by build tools to locate CUDA;
* add CUDA binaries such as `nvcc` to the system `PATH`;
* add CUDA libraries to `LD_LIBRARY_PATH`;
* reload the shell configuration;
* check that `nvcc` is available and points to the expected CUDA version.

## 3. Create the Python virtual environment

From the project root:

```bash
cd "$PROJECT_ROOT"

python3 -m venv venv
source venv/bin/activate

python -m pip install --upgrade pip setuptools wheel ninja
```

These commands:

* create a clean Python virtual environment;
* activate it;
* upgrade the basic Python packaging tools;
* install `ninja`, which is used during local compilation.

## 4. Install the project dependencies

The project dependencies are installed from the main requirements file:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file includes PyTorch with CUDA 12.8 wheels:

```text
torch==2.11.0+cu128
torchvision==0.26.0+cu128
torchaudio==2.11.0+cu128
```

After installation, PyTorch can be checked with:

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.get_arch_list())"
nvcc -V
```

These commands check:

* the installed PyTorch version;
* the CUDA version used by PyTorch;
* the GPU architectures supported by the installed PyTorch build;
* the CUDA Toolkit version available through `nvcc`.

## 5. Clone SegMamba locally

From the project root:

```bash
cd "$PROJECT_ROOT"
git clone https://github.com/ge-xing/SegMamba.git
```

The external SegMamba repository is cloned locally but is not included in this repository.

## 6. Apply the required manual changes

### Import in `src/models/segmamba.py`

In this project, the SegMamba import was adapted.

The original import:

```python
from mamba_ssm import Mamba
```

was replaced by:

```python
from mamba_ssm.modules.mamba_simple import Mamba
```

This change was required because the local `mamba-ssm` installation exposes the `Mamba` module from `mamba_ssm.modules.mamba_simple`.

### Support for `sm_120`

The following files from the local SegMamba repository were modified:

```text
SegMamba/causal-conv1d/setup.py
SegMamba/mamba/setup.py
```

The following block was added to include support for compute capability `sm_120`:

```python
if bare_metal_version >= Version("12.8"):
    cc_flag.append("-gencode")
    cc_flag.append("arch=compute_120,code=sm_120")
```

This change allows the CUDA extensions to be compiled for the NVIDIA RTX 5090.

## 7. Compile `causal-conv1d` locally

Before compiling, make sure CUDA 12.8 is available in the current shell:

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

Then compile and install `causal-conv1d` from the local SegMamba repository:

```bash
export CAUSAL_CONV1D_FORCE_BUILD=TRUE

cd "$PROJECT_ROOT/SegMamba/causal-conv1d"
pip install --no-build-isolation --no-cache-dir --no-deps --force-reinstall .
```

These commands:

* force local compilation instead of using a prebuilt wheel;
* compile the CUDA extension using the local CUDA Toolkit;
* install the package into the active virtual environment.

## 8. Compile `mamba-ssm` locally

```bash
export MAMBA_FORCE_BUILD=TRUE

cd "$PROJECT_ROOT/SegMamba/mamba"
pip install --no-build-isolation --no-cache-dir --no-deps --force-reinstall .
```

These commands:

* force local compilation of `mamba-ssm`;
* build it using the modified setup file with `sm_120` support;
* install it into the active virtual environment.

## 9. Final checks

From the project root:

```bash
cd "$PROJECT_ROOT"

python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.get_arch_list())"
python -c "import nibabel; print(nibabel.__version__)"
python -c "from mamba_ssm.modules.mamba_simple import Mamba; print('mamba-ssm OK')"
nvcc -V
```

These commands verify that:

* PyTorch is correctly installed;
* CUDA is available;
* the expected CUDA Toolkit is active;
* core medical imaging dependencies such as `nibabel` are installed;
* the local `mamba-ssm` installation can be imported.

After completing this setup, training and evaluation commands are described in the main `README.md`.
