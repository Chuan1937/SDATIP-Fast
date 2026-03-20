"""SDATIP: Fast Stochastic Determination of Arrival Time and Initial Polarity.

A high-performance Python package for seismic waveform analysis that determines
arrival time and initial polarity using Markov chain-based stochastic methods.

Example usage:
    >>> import sdatip
    >>> 
    >>> # Quick processing
    >>> result = sdatip.process_waveform(
    ...     name="station_A",
    ...     data=waveform_array,
    ...     output_dir="./output/"
    ... )
    >>> print(f"Arrival time: {result['arrival_time']:.3f}s")
    >>> print(f"Polarity: {result['polarity_up']:.3f}")
    >>> 
    >>> # Step-by-step processing
    >>> wf = sdatip.Waveform("station_A")
    >>> wf.importdata(data, delta=0.01)
    >>> wf.analyzedata()
    >>> # ... continue with other steps
"""

from sdatip.waveform import Waveform, findnoise
from sdatip.state import State
from sdatip.pmi import entropy, pmi, maxpmi, calculate_general
from sdatip.processor import process_waveform, process_batch

__version__ = "1.0.1"
__author__ = "He XingChen"

__all__ = [
    # Main classes
    "Waveform",
    "State",
    # High-level API
    "process_waveform",
    "process_batch",
    # Utility functions
    "findnoise",
    "entropy",
    "pmi",
    "maxpmi",
    "calculate_general",
]
