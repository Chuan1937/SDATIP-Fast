'''
Author: chuanjun
Date: 2025-05-27 18:57:42
email: chuanjun1978@gmail.com
website-github: https://github.com/Chuan1937
LastEditTime: 2025-05-27 23:15:06
'''
import numpy as np
import obspy
from single import solutionset
from tqdm import tqdm
import multiprocessing as mp
import os

MAX_LENGTH = 6000



def worker_wrap(args):
    i, trace, output_dir = args
    try:
        if len(trace.data) > MAX_LENGTH:
            trace.data = trace.data[:MAX_LENGTH]

        st1 = trace.slice(trace.stats.starttime, trace.stats.endtime)
        name = f"{st1.stats.station}"

        data = st1.data.astype(np.float32)

        solutionset(name, data, output_dir)
    except Exception as e:
        import traceback
        print(f"[Error] Trace {i} ({trace.id}) failed: {e}")
        traceback.print_exc() 


def main():


    datadir = '/home/chuan/下载/algorithm-withsac/algorithm/input/Hinettest/3.2-5/N.TKSH_onset.SAC'
    output_dir = './output/Hinettest/'


    NUM_WORKERS = 2


    os.makedirs(output_dir, exist_ok=True)

    print(f"Reading data from: {datadir}")
    stream = obspy.read(os.path.join(datadir, '*onset.SAC'))
    print(f"Found {len(stream)} traces to process.")

    args_list = [(i, trace, output_dir) for i, trace in enumerate(stream)]

    if NUM_WORKERS == -1:
        num_workers = mp.cpu_count()
        print(f"Configuration set to use all available cores: {num_workers}")
    elif NUM_WORKERS > mp.cpu_count():
        num_workers = mp.cpu_count()
        print(
            f"Warning: Requested workers ({NUM_WORKERS}) > available cores ({num_workers}). Using {num_workers} cores.")
    else:
        num_workers = NUM_WORKERS
        print(f"Configuration set to use {num_workers} worker(s).")

    if num_workers > 1:
        print("Running in multi-processing mode...")
        with mp.Pool(processes=num_workers) as pool:
            for _ in tqdm(pool.imap_unordered(worker_wrap, args_list), total=len(args_list), desc="Processing traces"):
                pass
    else:
        print("Running in single-threaded mode...")
        for item in tqdm(args_list, desc="Processing traces"):
            worker_wrap(item)
    print("Processing finished.")


if __name__ == '__main__':
    mp.freeze_support()
    main()
