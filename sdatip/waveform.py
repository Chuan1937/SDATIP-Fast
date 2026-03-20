"""Waveform preprocessing and analysis module.

Provides the Waveform class for seismic waveform data import, preprocessing,
interpolation, threshold analysis, and state construction for Markov modeling.
"""

import numpy as np
import obspy
from scipy.signal import argrelextrema

from sdatip.pmi import maxpmi
from sdatip.state import State


class Waveform:
    """Seismic waveform container with preprocessing and analysis capabilities.

    Handles data import, detrending, interpolation, threshold-based binning,
    and state construction for arrival time determination.
    """

    def __init__(self, name):
        self.name = name
        self.num = 1

    def importdata(self, data, delta):
        """Import waveform data directly from numpy array.

        Args:
            data: 1D numpy array of amplitude values.
            delta: Sampling interval in seconds.
        """
        self.data = data
        self.delta = delta
        self.length = len(self.data)
        self.timestamp = np.linspace(0, self.delta * (self.length - 1), self.length)

    def importdatafromsac(self, filename):
        """Import waveform data from SAC format file.

        Args:
            filename: Path to SAC file.
        """
        a = obspy.read("%s" % (filename))
        self.data = a[0].data
        self.delta = a[0].stats.delta
        self.length = len(self.data)
        self.timestamp = np.linspace(0, self.delta * (self.length - 1), self.length)

    def analyzedata(self):
        """Compute statistical properties and locate extreme points.

        Calculates mean, median, and identifies local maxima/minima.
        """
        self.mean = np.mean(self.data)
        self.median = np.median(self.data)
        self.extrememax = argrelextrema(self.data, np.greater_equal)[0]
        self.extrememin = argrelextrema(-1 * self.data, np.greater_equal)[0]
        self.extremeall = np.sort(np.append(self.extrememax, self.extrememin))
        self.above0 = len(np.where(self.data > 0)[0])
        self.below0 = len(np.where(self.data < 0)[0])

    def rmmean(self):
        """Remove mean from waveform data (detrend)."""
        self.data = self.data - self.mean
        self.above0 = len(np.where(self.data > 0)[0])
        self.below0 = len(np.where(self.data < 0)[0])

    def rmtail(self):
        """Remove trailing data after second-to-last extreme point."""
        self.data = self.data[0 : self.extremeall[-2] + 1]
        self.extremeall = self.extremeall[0:-1]
        self.length = len(self.data)
        self.extrememax = argrelextrema(self.data, np.greater_equal)[0]
        self.extrememin = argrelextrema(-1 * self.data, np.greater_equal)[0]
        self.above0 = len(np.where(self.data > 0)[0])
        self.below0 = len(np.where(self.data < 0)[0])

    def interpolate(self, coefficient):
        """Interpolate waveform to higher sampling density.

        Args:
            coefficient: Interpolation factor (e.g., 10 for 10x denser).
        """
        self.densedelta = self.delta / coefficient
        self.denselength = self.length + (self.length - 1) * (coefficient - 1)
        self.densetimestamp = np.linspace(0, self.densedelta * (self.denselength - 1), self.denselength)
        self.densedata = np.interp(self.densetimestamp, self.timestamp, self.data)
        self.denseextrememax = argrelextrema(self.densedata, np.greater_equal)[0]
        self.denseextrememin = argrelextrema(-1 * self.densedata, np.greater_equal)[0]
        self.denseextremeall = np.unique(np.append(self.denseextrememax, self.denseextrememin))
        self.denseco = coefficient
        self.dataindexindense = np.arange(0, self.denselength - 1 + self.denseco, self.denseco)

    def denseunique(self):
        """Extract unique threshold values from amplitude data.

        Sorts absolute values, groups indices by amplitude for threshold selection.
        """
        self.dense_abs = np.abs(self.densedata)
        sorter = np.argsort(self.dense_abs)
        self.dense_abs_unique, split_indices = np.unique(self.dense_abs[sorter], return_index=True)
        self.dense_abs_unique_index = np.split(sorter, split_indices[1:])
        self.threshold = self.dense_abs_unique

    def denselong(self, hvcoefficient, mininsertco):
        """Create dense time series with adaptive interpolation.

        Uses horizontal-vertical distance ratio to determine interpolation density.
        """
        if self.length <= 1:
            return
        hunit = hvcoefficient * (np.max(self.data) - np.min(self.data)) / (self.length - 1)
        vchange = np.abs(self.data[1:] - self.data[0:-1])
        insert_counts = np.round(np.sqrt(vchange**2 + hunit**2) * mininsertco / hunit).astype(int)
        timestamp_segments = [
            np.linspace(self.timestamp[i], self.timestamp[i + 1], insert_counts[i] + 1)[:-1]
            for i in range(self.length - 2)
        ]
        timestamp_segments.append(
            np.linspace(self.timestamp[-2], self.timestamp[-1], insert_counts[-1] + 1)
        )
        self.longtimestamp = np.concatenate(timestamp_segments)
        segment_lengths = insert_counts.copy()
        segment_lengths[-1] += 1
        start_indices = np.cumsum(np.concatenate(([0], segment_lengths)))
        self.dataindexinlong = np.append(start_indices[:-1], start_indices[-1] - 1).astype(int)
        self.denselongdata = np.interp(self.longtimestamp, self.timestamp, self.data)
        self.denselongabs = np.abs(self.denselongdata)
        self.longlength = len(self.denselongdata)
        self.longextrememax = argrelextrema(self.denselongdata, np.greater_equal)[0]
        self.longextrememin = argrelextrema(-1 * self.denselongdata, np.greater_equal)[0]
        self.longextremeall = np.unique(np.append(self.longextrememax, self.longextrememin))
        self.hvcoefficient = hvcoefficient
        self.mininsertco = mininsertco

    def extremearr(self):
        """Compute extreme point array with sign-based fill logic.

        Fills intervals between extreme points based on sign changes.
        """
        if len(self.longextremeall) < 2:
            self.longextremearr = np.zeros(self.longlength)
            self.longextremearrmin = 0
            return

        indices = self.longextremeall
        vals = self.denselongdata[indices]
        abs_vals = self.denselongabs[indices]
        is_same_sign = (vals[:-1] * vals[1:]) > 0
        self.longextremearr = np.zeros(self.longlength)
        same_sign_indices = np.where(is_same_sign)[0]

        if len(same_sign_indices) > 0:
            starts = indices[same_sign_indices]
            ends = indices[same_sign_indices + 1]
            fill_vals = np.maximum(abs_vals[same_sign_indices], abs_vals[same_sign_indices + 1])
            for start, end, val in zip(starts, ends, fill_vals):
                self.longextremearr[start : end + 1] = val
        diff_sign_indices = np.where(~is_same_sign)[0]
        for i in diff_sign_indices:
            start_idx = indices[i]
            end_idx = indices[i + 1]

            sub_array = self.denselongdata[start_idx : end_idx + 1]
            sign_change_products = sub_array[:-1] * sub_array[1:]
            try:
                turnpoint_relative_idx = np.where(sign_change_products <= 0)[0][0] + 1
            except IndexError:
                self.longextremearr[start_idx : end_idx + 1] = np.maximum(abs_vals[i], abs_vals[i + 1])
                continue
            turnpoint_abs_idx = start_idx + turnpoint_relative_idx
            self.longextremearr[start_idx:turnpoint_abs_idx] = abs_vals[i]
            self.longextremearr[turnpoint_abs_idx : end_idx + 1] = abs_vals[i + 1]
        self.longextremearrmin = np.min(self.longextremearr)

    def densebin(self):
        """Perform threshold-based binning analysis on dense waveform.

        Iterates through thresholds, computes PMI values, identifies cut points,
        and collects noise indices for state construction.
        """
        thresholds = self.threshold
        longlength = self.longlength
        longextremearr = self.longextremearr
        longextremearrmin = self.longextremearrmin
        denselongabs = self.denselongabs
        denselongdata = self.denselongdata
        longextremeall = self.longextremeall
        dataindexinlong = self.dataindexinlong
        dataindexindense = self.dataindexindense

        mivalue_list = []
        pmivalue_list = []
        cut_list = []
        peak_list = []
        noiselength_list = []
        noiseindex_list = []
        chances_list = []

        longarray = np.ones(longlength + 1)
        for i, threshold_i in enumerate(thresholds):
            longarray[:] = 1

            if threshold_i > longextremearrmin:
                zeroindex = np.where(np.abs(longextremearr) <= threshold_i)[0]
                longarray[zeroindex] = 0

                mivalue, pmivalue, longcutsolution = maxpmi(longarray, -1)

                for j, cut_val in enumerate(longcutsolution):
                    if cut_val < longlength:
                        try:
                            search_slice = denselongabs[cut_val + 1 :]
                            cutincycle = np.where(search_slice > threshold_i)[0][0]
                            longcutsolution[j] = cut_val + 1 + cutincycle
                        except IndexError:
                            longcutsolution[j] = longlength
            else:
                zeroindex = np.where(np.abs(denselongdata) <= threshold_i)[0]
                longarray[zeroindex] = 0
                mivalue, pmivalue, longcutsolution = maxpmi(longarray, -1)

            if threshold_i <= longextremearrmin:
                if len(zeroindex) > 0.95 * longlength:
                    mivalue, pmivalue, longcutsolution = 0, 0, np.array([longlength])
            else:
                zeroindex_check = np.where(np.abs(denselongdata) <= threshold_i)[0]
                if len(zeroindex_check) > 0.95 * longlength:
                    mivalue, pmivalue, longcutsolution = 0, 0, np.array([longlength])

            mivalue_list.append(mivalue)
            pmivalue_list.append(pmivalue)
            cut_list.append(longcutsolution - 1)

            peakcollect = []
            noiselength_inner = []
            noiseindex_inner = []
            chances_inner = []

            for cut_val in longcutsolution:
                oriindex = np.where(denselongabs[0 : int(cut_val)] <= threshold_i)[0]
                nindex, chance = findnoise(oriindex, dataindexinlong, dataindexindense, cut_val - 1)
                noiseindex_inner.append(nindex)
                chances_inner.append(chance)
                noiselength_inner.append(len(nindex) if hasattr(nindex, "__len__") else 0)

                if cut_val == longlength:
                    peakcollect.append(0)
                else:
                    try:
                        peak_idx_in_extremeall = np.searchsorted(longextremeall, cut_val - 1, side="right")

                        if peak_idx_in_extremeall >= len(longextremeall):
                            peakcollect.append(0)
                        else:
                            final_peak_idx = longextremeall[peak_idx_in_extremeall]
                            peakcollect.append(denselongdata[final_peak_idx])
                    except IndexError:
                        peakcollect.append(0)

            peak_list.append(peakcollect)
            noiselength_list.append(noiselength_inner)
            noiseindex_list.append(noiseindex_inner)
            chances_list.append(chances_inner)

        self.mivalue = mivalue_list
        self.pmivalue = pmivalue_list
        self.cut = cut_list
        self.peak = peak_list
        self.noiselength = noiselength_list
        self.noiseindex = noiseindex_list
        self.chances = chances_list

    def constructstate(self):
        """Build State model from processed threshold results.

        Groups noise indices by threshold and creates State objects
        for Markov chain analysis.
        """
        state = State(self.name)
        thresholds = self.threshold
        noise_indices = self.noiseindex
        densedata = self.densedata
        longtimestamp = self.longtimestamp

        for i in range(len(thresholds)):
            downthreshold = thresholds[i]
            upthreshold = thresholds[i + 1] if i < len(thresholds) - 1 else np.inf
            if len(self.noiselength[i]) > 1:
                all_indices_for_i = np.concatenate(noise_indices[i])
                noisecombo = densedata[all_indices_for_i.astype("int")]
                state.addcombo(
                    len(self.noiselength[i]),
                    downthreshold,
                    upthreshold,
                    noisecombo,
                    self.chances[i],
                    self.pmivalue[i],
                    self.peak[i],
                    longtimestamp[self.cut[i]],
                )
            else:
                if len(noise_indices[i]) > 0:
                    indices = noise_indices[i][0]
                    noise_values = densedata[indices.astype("int")]
                    state.addstate(
                        downthreshold,
                        upthreshold,
                        noise_values,
                        self.chances[i],
                        self.pmivalue[i],
                        self.peak[i],
                        longtimestamp[self.cut[i]],
                    )
        return state


def findnoise(oriindex, dataindexinlong, dataindexindense, cutsolution):
    """Find noise indices using vectorized binary search.

    Maps original indices to dense indices and returns the noise range
    along with the count of distribution segments.

    Args:
        oriindex: Original indices below threshold.
        dataindexinlong: Mapping from original to long array indices.
        dataindexindense: Mapping from original to dense array indices.
        cutsolution: Cut point index in long array.

    Returns:
        Tuple of (noise indices in dense array, number of segments).
    """
    if cutsolution == dataindexinlong[-1]:
        return np.arange(0, dataindexindense[-1] + 1), len(dataindexindense) - 1

    if len(oriindex) == 0:
        return np.array([]), 0

    bin_indices = np.searchsorted(dataindexinlong, oriindex, side="right") - 1

    distributionrange = np.unique(bin_indices)

    distributionrange = distributionrange[distributionrange >= 0]

    if len(distributionrange) == 0:
        return np.array([]), 0

    if len(distributionrange) == 1:
        segment_index = distributionrange[0]
        start = dataindexindense[segment_index]

        end = dataindexindense[segment_index + 1] if segment_index + 1 < len(dataindexindense) else dataindexindense[-1]

        return np.arange(start, end + 1), 1
    else:
        dense_starts = dataindexindense[distributionrange]
        dense_ends = dataindexindense[distributionrange + 1]

        all_ranges = [np.arange(start, end + 1) for start, end in zip(dense_starts, dense_ends)]

        noiseindex = np.concatenate(all_ranges)
        return np.unique(noiseindex), len(distributionrange)
