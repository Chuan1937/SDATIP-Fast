# SDATIP-Fast

Python implementation of SDATIP for fast stochastic determination of arrival time and initial polarity of seismic waveforms.

![Python](https://img.shields.io/badge/python-3.8+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Numba](https://img.shields.io/badge/numba-0.55+-red.svg)
![Numpy](https://img.shields.io/badge/numpy-1.20+-yellow.svg)
![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)

Python uses Numba JIT compilation optimization and vectorization, achieving speed improvements while maintaining complete consistency with the original stochastic algorithm.

## Performance

| Metric | Python+Numba | Original Python | Speedup |
|--------|-------------|---------|---------|
| 100Hz, 10s waveform | ~55 seconds | ~30 minutes | **~30x** |

## Accuracy Verification

**Key Results:**
- Arrival time estimation matches the original theoretical probability distribution completely.
- Initial polarity estimation maintains exactly the same mathematical rigor.
- Core logic produces identical pointwise mutual information matrices.

## Quick Start

```bash
pip install sdatip
```

### Basic Usage (Single Waveform Process)

```python
import sdatip
import obspy

# Read your waveform data (e.g., SAC file)
stream = obspy.read("Hinettest/N.AAKH_onset.SAC")
trace = stream[0]
data = trace.data         # 1D numpy array
delta = trace.stats.delta # Sampling interval in seconds

# Process the waveform
result = sdatip.process_waveform(
    name="example_station",
    data=data,
    output_dir="./output_example/",
    delta=delta,
    plot_enabled=False
)

print(f"Arrival time: {result['results'][0]['arrival_time']:.3f} s")
print(f"Polarity (up): {result['results'][0]['polarity_up']:.3f}")
```

### Batch Processing

```python
import sdatip

# Batch process all SAC files in a directory using multiprocessing
sdatip.process_batch(
    input_dir="Hinettest",
    output_dir="output_example",
    num_workers=4,    # Specify the number of CPU cores
    plot_enabled=True # Automatically generate probability visualization figures
)
```

## Features

- Stochastic determination of arrival time and initial polarity
- Markov chain probability modeling and density estimation
- Numba JIT compilation & vectorization for drastic speed optimization
- Built-in uncertainty and probability matrix visualization
- Multiprocessing batch support for handling massive waveform datasets seamlessly
- Core algorithm matches the original reference paper perfectly

## Documentation

See https://sdatip.readthedocs.io/ for full documentation including:
- API reference
- Algorithm details
- Output file specifications
- Performance optimization

## Run Tests

```bash
jupyter notebook example.ipynb
```

## Project Structure

```text
SDATIP-Fast/
├── sdatip/
│   ├── processor.py   # High-level API drivers (process_waveform, process_batch)
│   ├── waveform.py    # Preprocessing, extreme points, interpolations (Waveform)
│   ├── state.py       # Markov transition matrices, likelihood modeling (State)
│   ├── pmi.py         # Pointwise mutual information tools (PMI, MaxPMI)
│   └── plotting.py    # Display rendering and visualization logic
│
├── Hinettest/         # Example dataset containing SAC waveform files
└── example.ipynb      # Complete test script & interactive demonstration
```

## Author

He XingChen

## License

MIT License

## References

Pei, W., Zhuang, J. & Zhou, S. Stochastic determination of arrival time and initial polarity of seismic waveform. *Earth Planets Space* 77, 36 (2025). https://doi.org/10.1186/s40623-025-02161-5
