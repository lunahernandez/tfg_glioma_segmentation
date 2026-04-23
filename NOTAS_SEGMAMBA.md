- CUDA Toolkit usado: 12.8
- PyTorch: 2.11.0+cu128
- transformers: 4.36.2
- Se modificó SegMamba/causal-conv1d/setup.py añadiendo sm_120
- Se modificó SegMamba/mamba/setup.py añadiendo sm_120
- Import en segmamba.py:
  from mamba_ssm.modules.mamba_simple import Mamba
