# Usage

This guide provides examples on how to use `SDATIP-Fast` to determine the arrival time and initial polarity of seismic waveforms.

## Single Waveform Processing

For a single waveform, you can extract the data array and execute the processor:

```python
import sdatip
import obspy

# 1. Read waveform data
stream = obspy.read("data/N.AAKH_onset.SAC")
trace = stream[0]
data = trace.data
delta = trace.stats.delta

# 2. Process waveform
result = sdatip.process_waveform(
    name="N.AAKH",
    data=data,
    output_dir="./output_example/",
    delta=delta,
    plot_enabled=False
)

# 3. View Results
print(f"Estimated Arrival Time: {result['results'][0]['arrival_time']:.3f} s")
print(f"Polarity (up): {result['results'][0]['polarity_up']:.3f}")
```

## Batch Processing with Multiprocessing

SDATIP includes built-in parallelization capabilities using multiple CPU cores to evaluate large directories of `.SAC` files:

```python
import sdatip

# Run the batch processes
sdatip.process_batch(
    input_dir="Hinettest",
    output_dir="output_example",
    num_workers=4,    # Specifies 4 cores 
    plot_enabled=True # Automatically generate PDF plots
)
```

## Advanced Step-by-Step State Processing

If you need finer control over the internal state logic or variables:

```python
import sdatip

# Initialize object
wf = sdatip.Waveform("station")
wf.importdata(data, delta=0.01)

# Preprocessing
wf.analyzedata()
wf.rmmean()
wf.interpolate(1)
wf.denseunique()
wf.denselong(hvcoefficient=2.5, mininsertco=200)
wf.extremearr()
wf.densebin()

# Core Markov State Matrix
state = wf.constructstate()
timeprobs, num_solutions = state.markovmatrix()
ampprob = state.ampprobcalculate()

# Estimate target parameter
state.estimation(0)
print(f"Arrival: {state.arrivalestimate:.3f} s")
print(f"Polarity up: {state.polarityup:.3f}")
```