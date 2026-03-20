# SDATIP

[![PyPI version](https://badge.fury.io/py/sdatip.svg)](https://badge.fury.io/py/sdatip)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Fast Stochastic Determination of Arrival Time and Initial Polarity of Seismic Waveforms**

A high-performance Python package for seismic waveform analysis that determines arrival time and initial polarity using Markov chain-based stochastic methods.

## Performance

| Metric | Original | SDATIP |
|--------|----------|--------|
| 100Hz, 10s waveform | ~30 minutes | ~55 seconds |
| Speedup | - | **30x faster** |

## Installation

```bash
pip install sdatip
```

Or install from source:

```bash
git clone https://github.com/Chuan1937/SDATIP-Fast.git
cd SDATIP-Fast
pip install -e .
```

## Quick Start

### Process a Single Waveform

```python
import numpy as np
import sdatip

# Your waveform data (1D numpy array)
data = np.random.randn(1000)

# Process the waveform
result = sdatip.process_waveform(
    name="STATION_A",
    data=data,
    output_dir="./output/"
)

# Get results
print(f"Arrival time: {result['results'][0]['arrival_time']:.3f}s")
print(f"Polarity (up): {result['results'][0]['polarity_up']:.3f}")
```

### Process Multiple Waveforms

```python
import sdatip

# Batch process all SAC files in a directory
results = sdatip.process_batch(
    input_dir="./data/",
    output_dir="./output/",
    num_workers=4,  # Use 4 CPU cores
    plot_enabled=False
)
```

### Step-by-Step Processing

```python
import sdatip

# Create waveform object
wf = sdatip.Waveform("station")
wf.importdata(data, delta=0.01)
wf.analyzedata()
wf.rmmean()
wf.interpolate(1)
wf.denseunique()
wf.denselong(hvcoefficient=2.5, mininsertco=200)
wf.extremearr()
wf.densebin()

# Build state model
state = wf.constructstate()
timeprobs, num_solutions = state.markovmatrix()
ampprob = state.ampprobcalculate()

# Estimate results
state.estimation(0)
print(f"Arrival: {state.arrivalestimate:.3f}s")
print(f"Polarity up: {state.polarityup:.3f}")
```

## API Reference

### `process_waveform(name, data, output_dir, ...)`

Process a single seismic waveform.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | required | Station name |
| `data` | np.ndarray | required | 1D amplitude array |
| `output_dir` | str | required | Output directory |
| `delta` | float | 0.01 | Sampling interval (s) |
| `plot_enabled` | bool | True | Generate plots |

### `process_batch(input_dir, output_dir, ...)`

Process multiple waveforms in parallel.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_dir` | str | required | Input directory with SAC files |
| `output_dir` | str | required | Output directory |
| `num_workers` | int | -1 | CPU cores (-1 = all) |
| `plot_enabled` | bool | False | Generate plots |

### Classes

- `Waveform(name)` - Waveform preprocessing container
- `State(name)` - Markov state model

## Output Files

| File | Description |
|------|-------------|
| `{station}.txt` | Text summary with arrival time and polarity |
| `{station}.npz` | Main results (matrix, probabilities, thresholds) |
| `{station}_timeprob_{i}.npz` | Time probability distribution |
| `{station}_{i}.pdf` | Probability plot (if enabled) |
| `{station}.eps` | Publication-ready plot (if enabled) |

## Algorithm

Based on: Pei, W., Zhuang, J. & Zhou, S. "Stochastic determination of arrival time and initial polarity of seismic waveform." *Earth Planets Space* 77, 36 (2025). https://doi.org/10.1186/s40623-025-02161-5

## Validation

```bash
python benchmark_validate.py check
```

## Citation

```bibtex
@article{pei2025stochastic,
  title={Stochastic determination of arrival time and initial polarity of seismic waveform},
  author={Pei, W. and Zhuang, J. and Zhou, S.},
  journal={Earth, Planets and Space},
  volume={77},
  pages={36},
  year={2025},
  doi={10.1186/s40623-025-02161-5}
}
```

## License

MIT License

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Chuan1937/SDATIP-Fast&type=Date)](https://star-history.com/#Chuan1937/SDATIP-Fast&Date)
