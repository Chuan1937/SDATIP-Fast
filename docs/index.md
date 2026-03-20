# SDATIP-Fast Documentation

Welcome to the documentation for **SDATIP-Fast**!

Python implementation of SDATIP for fast stochastic determination of arrival time and initial polarity of seismic waveforms.

Python uses Numba JIT compilation optimization and vectorization, achieving speed improvements while maintaining complete consistency with the original stochastic algorithm.

## Overview

SDATIP-Fast is a high-performance Python package for seismic waveform analysis that determines arrival time and initial polarity using Markov chain-based stochastic methods.

### Key Features
- **High Performance**: ~30x faster than original Python implementation using Numba JIT.
- **Accurate**: Exact algorithmic compliance with the original stochastic method by Pei et al. (2025).
- **Multiprocessing**: Built-in support for processing massive databases in parallel.
- **Visualization**: Generates rich matrix and probability plots for uncertainty validation.

## Table of Contents

```{toctree}
:maxdepth: 2

installation
usage
api
```