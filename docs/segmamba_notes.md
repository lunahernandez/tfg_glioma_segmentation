# Notas de instalación de SegMamba

Este documento describe la configuración adicional necesaria para ejecutar los experimentos basados en SegMamba dentro de este proyecto.

SegMamba se instaló localmente a partir del repositorio original. Debido a problemas de compatibilidad con la NVIDIA RTX 5090, las dependencias `causal-conv1d` y `mamba-ssm` tuvieron que compilarse manualmente con soporte para la capacidad de cómputo `sm_120`.

## Entorno utilizado

* Sistema operativo: WSL Ubuntu
* GPU: NVIDIA RTX 5090
* CUDA Toolkit: 12.8
* PyTorch: 2.11.0+cu128
* MONAI: 1.5.2
* transformers: 4.36.2

## Nota importante

`causal-conv1d` y `mamba-ssm` no se incluyen en `requirements.txt`, ya que deben instalarse localmente desde el repositorio de SegMamba después de aplicar los cambios de compatibilidad descritos en este documento.

En los comandos siguientes, sustituir `/path/to/tfg_glioma_segmentation` por la ruta local en la que se haya clonado este repositorio.

```bash
export PROJECT_ROOT=/path/to/tfg_glioma_segmentation
```

## 1. Instalar CUDA Toolkit 12.8 en WSL

SegMamba requiere la compilación local de paquetes dependientes de CUDA. Por este motivo, se utilizó CUDA Toolkit 12.8 bajo WSL.

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

Estos comandos:

* añaden el repositorio de paquetes de NVIDIA CUDA para WSL;
* registran la clave de firma del repositorio;
* actualizan el índice local de paquetes;
* instalan CUDA Toolkit 12.8, incluyendo `nvcc`, bibliotecas CUDA y archivos de cabecera.

## 2. Configurar CUDA 12.8 como versión activa

```bash
sudo ln -sfn /usr/local/cuda-12.8 /usr/local/cuda

echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc

source ~/.bashrc
hash -r
nvcc -V
```

Estos comandos:

* crean `/usr/local/cuda` como enlace simbólico a CUDA 12.8;
* definen `CUDA_HOME`, utilizado por las herramientas de compilación para localizar CUDA;
* añaden binarios de CUDA, como `nvcc`, al `PATH` del sistema;
* añaden las bibliotecas de CUDA a `LD_LIBRARY_PATH`;
* recargan la configuración de la terminal;
* comprueban que `nvcc` está disponible y apunta a la versión esperada de CUDA.

## 3. Crear el entorno virtual de Python

Desde la raíz del proyecto:

```bash
cd "$PROJECT_ROOT"

python3 -m venv venv
source venv/bin/activate

python -m pip install --upgrade pip wheel ninja
python -m pip install "setuptools==70.2.0"
```

Estos comandos:

* crean un entorno virtual limpio;
* activan el entorno virtual;
* actualizan las herramientas básicas de empaquetado;
* instalan `ninja`, utilizado durante la compilación local;
* instalan la versión de `setuptools` utilizada en el entorno que funcionó correctamente.

## 4. Instalar las dependencias del proyecto

Las dependencias generales del proyecto se instalan desde el archivo principal de requisitos:

```bash
pip install -r requirements.txt
```

El archivo `requirements.txt` incluye la versión de PyTorch con soporte para CUDA 12.8:

```text
torch==2.11.0+cu128
torchvision==0.26.0+cu128
torchaudio==2.11.0+cu128
```

Después de la instalación, PyTorch puede comprobarse con:

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.get_arch_list())"
nvcc -V
```

Estos comandos comprueban:

* la versión instalada de PyTorch;
* la versión de CUDA utilizada por PyTorch;
* las arquitecturas de GPU soportadas por la instalación de PyTorch;
* la versión de CUDA Toolkit disponible mediante `nvcc`.

## 5. Clonar SegMamba localmente

Desde la raíz del proyecto:

```bash
cd "$PROJECT_ROOT"
git clone https://github.com/ge-xing/SegMamba.git
```

El repositorio externo de SegMamba se clona localmente, pero no se incluye dentro de este repositorio.

## 6. Adaptar la implementación de SegMamba

La implementación del modelo SegMamba utilizada en este proyecto se adaptó a partir del archivo del repositorio público:

```text
SegMamba/model_segmamba/segmamba.py
```

Ese archivo se copió y adaptó en el proyecto como:

```text
src/models/segmamba.py
```

Esta adaptación permite integrar SegMamba en el mismo flujo experimental que el resto de modelos evaluados.

### Cambio del import de Mamba

En `src/models/segmamba.py`, el import original:

```python
from mamba_ssm import Mamba
```

se sustituyó por:

```python
from mamba_ssm.modules.mamba_simple import Mamba
```

Este cambio fue necesario porque la instalación local de `mamba-ssm` expone el módulo `Mamba` desde `mamba_ssm.modules.mamba_simple`.

### Soporte para `sm_120`

También fue necesario modificar los siguientes archivos del repositorio local de SegMamba:

```text
SegMamba/causal-conv1d/setup.py
SegMamba/mamba/setup.py
```

En ambos archivos se añadió el siguiente bloque para incluir soporte para la capacidad de cómputo `sm_120`:

```python
if bare_metal_version >= Version("12.8"):
    cc_flag.append("-gencode")
    cc_flag.append("arch=compute_120,code=sm_120")
```

Este cambio permite compilar las extensiones CUDA para la NVIDIA RTX 5090.

## 7. Compilar `causal-conv1d` localmente

Antes de compilar, comprobar que CUDA 12.8 está disponible en la terminal actual:

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

Después, compilar e instalar `causal-conv1d` desde el repositorio local de SegMamba:

```bash
export CAUSAL_CONV1D_FORCE_BUILD=TRUE

cd "$PROJECT_ROOT/SegMamba/causal-conv1d"
pip install --no-build-isolation --no-cache-dir --no-deps --force-reinstall .
```

Estos comandos:

* fuerzan la compilación local en lugar de utilizar una rueda precompilada;
* compilan la extensión CUDA utilizando el CUDA Toolkit local;
* instalan el paquete en el entorno virtual activo.

## 8. Compilar `mamba-ssm` localmente

```bash
export MAMBA_FORCE_BUILD=TRUE

cd "$PROJECT_ROOT/SegMamba/mamba"
pip install --no-build-isolation --no-cache-dir --no-deps --force-reinstall .
```

Estos comandos:

* fuerzan la compilación local de `mamba-ssm`;
* compilan el paquete utilizando el archivo `setup.py` modificado con soporte para `sm_120`;
* instalan el paquete en el entorno virtual activo.

## 9. Comprobaciones finales

Desde la raíz del proyecto:

```bash
cd "$PROJECT_ROOT"

python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.get_arch_list())"
python -c "import nibabel; print(nibabel.__version__)"
python -c "from mamba_ssm.modules.mamba_simple import Mamba; print('mamba-ssm OK')"
nvcc -V
```

Estos comandos verifican que:

* PyTorch está correctamente instalado;
* CUDA está disponible;
* CUDA Toolkit apunta a la versión esperada;
* dependencias de imagen médica como `nibabel` están instaladas;
* la instalación local de `mamba-ssm` puede importarse correctamente.

Una vez completada esta configuración, los comandos de entrenamiento y evaluación se describen en el `README.md` principal.



