"""SDATIP-Fast: Optimized seismic waveform arrival time and polarity analysis.

This is a high-performance implementation of the algorithm from the paper:
"Stochastic determination of arrival time and initial polarity of seismic waveform"

Usage:
    Edit the configuration variables at the top of this file, then run:
        python main.py
"""

import numpy as np
import obspy
from single import solutionset
from tqdm import tqdm
import multiprocessing as mp
import os

MAX_LENGTH = 6000

# ==================== Configuration ====================
INPUT_DIR = "./Hinettest/"
OUTPUT_DIR = "./output/Hinettest/"
NUM_WORKERS = -1          # -1 = use all available CPU cores
PLOT_ENABLED = False     # False = skip plotting to save 5-10% time
# ====================================================


def worker_wrap(args):
    """Worker function to process a single seismic trace."""
    i, trace, output_dir, plot_enabled = args
    try:
        if len(trace.data) > MAX_LENGTH:
            trace.data = trace.data[:MAX_LENGTH]

        st1 = trace.slice(trace.stats.starttime, trace.stats.endtime)
        name = f"{st1.stats.station}"

        data = st1.data.astype(np.float32)

        solutionset(name, data, output_dir, plot_enabled)
    except Exception as e:
        import traceback

        print(f"[Error] Trace {i} ({trace.id}) failed: {e}")
        traceback.print_exc()


def main():
    datadir = INPUT_DIR
    output_dir = OUTPUT_DIR
    num_workers_config = NUM_WORKERS
    plot_enabled = PLOT_ENABLED

    os.makedirs(output_dir, exist_ok=True)

    print(f"Reading data from: {datadir}")
    stream = obspy.read(os.path.join(datadir, "*onset.SAC"))
    print(f"Found {len(stream)} traces to process.")

    if not plot_enabled:
        print("[INFO] Plotting disabled (PLOT_ENABLED = False)")

    args_list = [(i, trace, output_dir, plot_enabled) for i, trace in enumerate(stream)]

    if num_workers_config == -1:
        num_workers = mp.cpu_count()
        print(f"Configuration set to use all available cores: {num_workers}")
    elif num_workers_config > mp.cpu_count():
        num_workers = mp.cpu_count()
        print(
            f"Warning: Requested workers ({num_workers_config}) > available cores ({num_workers}). "
            f"Using {num_workers} cores."
        )
    else:
        num_workers = num_workers_config
        print(f"Configuration set to use {num_workers} worker(s).")

    if num_workers > 1:
        print("Running in multi-processing mode...")
        with mp.Pool(processes=num_workers) as pool:
            for _ in tqdm(
                pool.imap_unordered(worker_wrap, args_list),
                total=len(args_list),
                desc="Processing traces",
            ):
                pass
    else:
        print("Running in single-threaded mode...")
        for item in tqdm(args_list, desc="Processing traces"):
            worker_wrap(item)
    print("Processing finished.")


if __name__ == "__main__":
    mp.freeze_support()
    main()
