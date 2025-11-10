# SDATIP-Fast
Optimized version of the algorithm implemented in the paper “Stochastic determination of arrival time and initial polarity of seismic waveform”.

The original paper algorithm is open-sourced at https:// doi.org/https://doi.org/10.5281/zenodo.13918012, but it is very slow, taking close to 30 minutes just for a 100hz, 10s waveform. I optimized it, using numba, vectorization, binary lookup and other optimization algorithms to greatly improve its speed, so that it can handle 100hz, 10s waveforms need only 40s. and can use multi-threaded and single-threaded mode.
# Usage
Inside Hinettest is the test dataset, replace its path into main.py, and then adjust NUM_WORKERS according to the number of cores of CPU to run directly.
# Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Chuan1937/SDATIP-Fast&type=Date)](https://star-history.com/#Chuan1937/SDATIP-Fast&Date)

# Reference
Pei, W., Zhuang, J. & Zhou, S. Stochastic determination of arrival time and initial polarity of seismic waveform. Earth Planets Space 77, 36 (2025).
https://doi.org/10.1186/s40623-025-02161-5
