"""Pointwise Mutual Information (PMI) calculations for seismic waveform analysis.

This module provides JIT-compiled functions for computing PMI and entropy
to find optimal cut points in binary sequences.
"""

import numba
import numpy as np


@numba.jit(nopython=True, cache=True)
def entropy(ampen2):
    """Calculate binary entropy of a 0/1 sequence.

    Uses Shannon entropy formula: H = -p0*log2(p0) - p1*log2(p1)
    """
    if ampen2.size == 0:
        return 0.0
    counts = np.bincount(ampen2)
    if len(counts) < 2:
        return 0.0
    zero_count = counts[0]
    one_count = counts[1]
    if zero_count == 0 or one_count == 0:
        return 0.0
    length = len(ampen2)
    p_zero = zero_count / length
    p_one = one_count / length
    return -(np.log2(p_zero) * p_zero + np.log2(p_one) * p_one)


@numba.jit(nopython=True, cache=True)
def pmi(ampbi, t):
    """Calculate Pointwise Mutual Information at split point t.

    Computes PMI matrix between two parts of binary sequence split at index t.
    Returns (mi_value, normalized_pmi_value).
    """
    length_all = len(ampbi)
    part1 = ampbi[:t]
    part2 = ampbi[t:]

    zero_before_split = (part1 == 0).sum()
    one_before_split = (part1 == 1).sum()
    zero_after_split = (part2 == 0).sum()
    one_after_split = (part2 == 1).sum()

    pmi_matrix = np.zeros((2, 2))
    npmi_matrix = np.zeros((2, 2))

    total_zeros = zero_before_split + zero_after_split
    total_ones = one_before_split + one_after_split
    len_part1 = zero_before_split + one_before_split
    len_part2 = zero_after_split + one_after_split

    if zero_before_split > 0 and total_zeros > 0 and len_part1 > 0:
        denominator = len_part1 * total_zeros
        pmi_val = zero_before_split * length_all / denominator
        pmi_matrix[0, 0] = (zero_before_split / length_all) * np.log2(pmi_val)
        npmi_matrix[0, 0] = -np.log2(pmi_val) / np.log2(zero_before_split / length_all)
    else:
        npmi_matrix[0, 0] = -1.0

    if one_before_split > 0 and total_ones > 0 and len_part1 > 0:
        denominator = len_part1 * total_ones
        pmi_val = one_before_split * length_all / denominator
        pmi_matrix[0, 1] = (one_before_split / length_all) * np.log2(pmi_val)
        npmi_matrix[0, 1] = -np.log2(pmi_val) / np.log2(one_before_split / length_all)
    else:
        npmi_matrix[0, 1] = -1.0

    if zero_after_split > 0 and total_zeros > 0 and len_part2 > 0:
        denominator = len_part2 * total_zeros
        pmi_val = zero_after_split * length_all / denominator
        pmi_matrix[1, 0] = (zero_after_split / length_all) * np.log2(pmi_val)
        npmi_matrix[1, 0] = -np.log2(pmi_val) / np.log2(zero_after_split / length_all)
    else:
        npmi_matrix[1, 0] = -1.0

    if one_after_split > 0 and total_ones > 0 and len_part2 > 0:
        denominator = len_part2 * total_ones
        pmi_val = one_after_split * length_all / denominator
        pmi_matrix[1, 1] = (one_after_split / length_all) * np.log2(pmi_val)
        npmi_matrix[1, 1] = -np.log2(pmi_val) / np.log2(one_after_split / length_all)
    else:
        npmi_matrix[1, 1] = -1.0

    mi_value = pmi_matrix.sum()
    normalized_pmi = (
        npmi_matrix[0, 0] * zero_before_split
        - npmi_matrix[0, 1] * one_before_split
        - npmi_matrix[1, 0] * zero_after_split
        + npmi_matrix[1, 1] * one_after_split
    ) / length_all
    return mi_value, normalized_pmi


@numba.jit(nopython=True, cache=True)
def maxpmi(ampbi, n):
    """Find the split point that maximizes PMI.

    Args:
        ampbi: Binary sequence (0s and 1s)
        n: If -1, search all transition points; otherwise use specified index

    Returns:
        (max_mi, max_normalized_pmi, split_indices)
    """
    zero_count = (ampbi == 0).sum()
    one_count = (ampbi == 1).sum()

    if zero_count == 0 or one_count == 0:
        return -1.0, -1.0, np.array([-1], dtype=np.int64)

    if n != -1:
        mi_val, normalized_pmi_val = pmi(ampbi, n)
        return mi_val, normalized_pmi_val, np.array([n], dtype=np.int64)

    transition_indices = np.where(ampbi[1:] - ampbi[:-1] == 1)[0]
    if len(transition_indices) == 0:
        return -1.0, -1.0, np.array([-1], dtype=np.int64)

    results = np.zeros((len(transition_indices), 2))
    for i in range(len(transition_indices)):
        results[i, 0], results[i, 1] = pmi(ampbi, transition_indices[i] + 1)

    max_mi_value = np.max(results[:, 0])
    best_indices = np.where(results[:, 0] == max_mi_value)[0]

    final_split_points = transition_indices[best_indices] + 1
    return max_mi_value, results[best_indices[0], 1], final_split_points


@numba.jit(nopython=True, cache=True)
def calculate_general(xsquare, n):
    """Calculate general form for n-dimensional Gaussian integral.

    Uses the formula for computing integrals of x^n * exp(-x^2/2*sigma^2).
    """
    if n == 3:
        return 1.0 / xsquare
    if n == 2:
        return np.sqrt(np.pi / 2.0 / xsquare)
    if n % 2 == 1:
        m = (n - 3) / 2.0
        prod = 1.0
        for val in range(2, n - 1, 2):
            prod *= val
        return prod * (1.0 / xsquare) ** (m + 1.0)
    else:
        m = (n - 2) / 2.0
        prod = 1.0
        for val in range(1, n - 2, 2):
            prod *= val
        return prod * np.sqrt(np.pi / 2.0 / xsquare) * (1.0 / xsquare) ** m
