# Installation

SDATIP-Fast can be installed from PyPI or built from source to fit your environment.

## Prerequisites

- Python 3.8 or higher
- `numpy`
- `numba`
- `obspy` (for handling SAC files)
- `scipy`
- `matplotlib`
- `tqdm`

## Install via pip

The easiest way to install SDATIP is via the Python Package Index (PyPI):

```bash
pip install sdatip
```

## Install from Source

If you want the latest development version or wish to contribute:

```bash
git clone https://github.com/Chuan1937/SDATIP-Fast.git
cd SDATIP-Fast
pip install -e .
```

## Verify Installation

You can check if the installation was successful by running a simple import:

```python
import sdatip
print(sdatip.__version__)
```