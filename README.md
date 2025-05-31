# SDATIP-Fast
Optimized version of the algorithm implemented in the paper “Stochastic determination of arrival time and initial polarity of seismic waveform”.

The original paper algorithm is open-sourced at https:// doi.org/https://doi.org/10.5281/zenodo.13918012, but it is very slow, taking close to 30 minutes just for a 100hz, 10s waveform. I optimized it, using numba, vectorization, binary lookup and other optimization algorithms to greatly improve its speed, so that it can handle 100hz, 10s waveforms need only 40s. and can use multi-threaded and single-threaded mode.

