"""High-level processing functions for SDATIP.

This module provides convenient functions for processing seismic waveforms
without dealing with low-level details.
"""

import os
from typing import Optional

import numpy as np

from sdatip.waveform import Waveform
from sdatip.state import State


def process_waveform(
    name: str,
    data: np.ndarray,
    output_dir: str,
    delta: float = 0.01,
    hvcoefficient: float = 2.5,
    mininsertco: int = 200,
    plot_enabled: bool = True,
    max_length: Optional[int] = None,
) -> dict:
    """Process a single seismic waveform and estimate arrival time and polarity.

    This is the main entry point for processing a single waveform. It handles
    all steps from data import to final estimation.

    Args:
        name: Station name or identifier.
        data: 1D numpy array of amplitude values.
        output_dir: Directory to save output files.
        delta: Sampling interval in seconds (default: 0.01 for 100Hz).
        hvcoefficient: Horizontal-vertical distance ratio for dense interpolation.
        mininsertco: Minimum insertion coefficient for dense interpolation.
        plot_enabled: Whether to generate visualization plots.
        max_length: Optional maximum length to truncate data.

    Returns:
        Dictionary containing:
            - arrival_time: Estimated arrival time in seconds
            - polarity_up: Probability of upward polarity
            - polarity_down: Probability of downward polarity
            - polarity_unknown: Probability of unknown polarity
            - num_solutions: Number of solutions found
            - state: State object for further analysis
            - waveform: Waveform object for further analysis

    Example:
        >>> import numpy as np
        >>> import sdatip
        >>> data = np.random.randn(1000)  # Your waveform data
        >>> result = sdatip.process_waveform(
        ...     name="STATION_A",
        ...     data=data,
        ...     output_dir="./output/"
        ... )
        >>> print(f"Arrival: {result['arrival_time']:.3f}s")
    """
    os.makedirs(output_dir, exist_ok=True)

    if max_length is not None and len(data) > max_length:
        data = data[:max_length]

    wf = Waveform(name)
    wf.importdata(data, delta)
    wf.analyzedata()
    wf.interpolate(1)
    wf.denseunique()
    wf.denselong(hvcoefficient, mininsertco)
    wf.extremearr()
    wf.densebin()

    state = wf.constructstate()
    timeprobs, num_solutions = state.markovmatrix()
    ampprob = state.ampprobcalculate()

    results = []

    for i in range(num_solutions):
        state.estimation(i)

        result = {
            "solution_id": i,
            "arrival_time": state.arrivalestimate,
            "polarity_up": state.polarityup,
            "polarity_down": state.polaritydown,
            "polarity_unknown": state.polarityunknown,
            "A_peak": state.Apeakestimate,
            "sigma": state.sigmaestimate,
        }
        results.append(result)

        main_data = {
            "transitionmatrix": np.array(state.matrix),
            "ampprob": np.array(state.ampprob_up).astype("float64"),
            "Apeak": state.Apeak,
            "samplelength": state.samplength,
            "eigvalue": state.eigvalue,
            "bigeig": state.bigeig,
            "threshold": wf.threshold,
        }

        main_filepath = os.path.join(output_dir, f"{name}.npz")
        np.savez_compressed(main_filepath, **main_data, allow_pickle=True)

        timeprob_filepath = os.path.join(output_dir, f"{name}_timeprob_{i}.npz")
        np.savez_compressed(timeprob_filepath, timeprob=timeprobs[i])

        txt_filepath = os.path.join(output_dir, f"{name}.txt")
        with open(txt_filepath, "a") as f:
            line = (
                f"{name} solution id:{i} "
                f"arrivaltime:{state.arrivalestimate:.3f} "
                f"overall up:{float(np.sum(timeprobs[i] * ampprob)):.5f} "
                f"up:{state.polarityup:.3f} "
                f"down:{state.polaritydown:.3f} "
                f"unknown:{state.polarityunknown:.3f}\n"
            )
            f.write(line)

        if plot_enabled:
            from sdatip.plotting import plot_result, plot_result_graduate

            plot_result(wf, state, i, name, output_dir)
            plot_result_graduate(wf, state, i, name, output_dir)

    return {
        "results": results,
        "state": state,
        "waveform": wf,
        "num_solutions": num_solutions,
    }


def process_batch(
    input_dir: str,
    output_dir: str,
    num_workers: int = -1,
    plot_enabled: bool = False,
    pattern: str = "*onset.SAC",
    max_length: int = 6000,
) -> list:
    """Process multiple seismic waveforms in parallel.

    Args:
        input_dir: Directory containing SAC files.
        output_dir: Directory to save output files.
        num_workers: Number of parallel workers (-1 for all cores).
        plot_enabled: Whether to generate plots (slower but produces visualizations).
        pattern: Glob pattern for input files.
        max_length: Maximum waveform length to process.

    Returns:
        List of results from each waveform.

    Example:
        >>> import sdatip
        >>> results = sdatip.process_batch(
        ...     input_dir="./data/",
        ...     output_dir="./output/",
        ...     num_workers=4
        ... )
    """
    import multiprocessing as mp
    from glob import glob

    import obspy
    from tqdm import tqdm

    os.makedirs(output_dir, exist_ok=True)

    files = sorted(glob(os.path.join(input_dir, pattern)))
    if not files:
        raise ValueError(f"No files found matching pattern '{pattern}' in {input_dir}")

    stream = obspy.read(os.path.join(input_dir, pattern))
    print(f"Found {len(stream)} traces to process.")

    if num_workers == -1:
        num_workers = mp.cpu_count()
    num_workers = min(num_workers, mp.cpu_count())

    if num_workers > len(stream):
        num_workers = len(stream)

    def process_trace(args):
        idx, trace = args
        try:
            data = trace.data.astype(np.float32)
            if len(data) > max_length:
                data = data[:max_length]

            return process_waveform(
                name=trace.stats.station,
                data=data,
                output_dir=output_dir,
                plot_enabled=plot_enabled,
            )
        except Exception as e:
            print(f"[Error] Trace {idx} ({trace.id}) failed: {e}")
            return None

    args_list = list(enumerate(stream))

    if num_workers > 1:
        print(f"Processing with {num_workers} workers...")
        with mp.Pool(num_workers) as pool:
            results = list(tqdm(pool.imap(process_trace, args_list), total=len(args_list)))
    else:
        print("Processing in single-threaded mode...")
        results = [process_trace(args) for args in tqdm(args_list)]

    results = [r for r in results if r is not None]
    print(f"Successfully processed {len(results)} traces.")

    return results
