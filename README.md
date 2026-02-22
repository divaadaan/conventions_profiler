# conventions-profiler

## Installation

Install PyTorch with CUDA separately before installing this package — it cannot be included in `requirements.txt` because it requires a CUDA-specific index URL.

Go to [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally), select your platform and CUDA version, and run the generated command inside your virtual environment. Example for CUDA 12.4:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Then install this package:

```bash
pip install -e ".[dev]"
```
