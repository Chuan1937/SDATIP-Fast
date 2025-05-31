import numba
import numpy as np


@numba.jit(nopython=True)
def Entropy(ampen2):
    """
    - Numba JIT 加速。
    """
    if ampen2.size == 0:
        return 0.0
    counts = np.bincount(ampen2)
    if len(counts) < 2:
        return 0.0
    zeronum = counts[0]
    onenum = counts[1]
    if zeronum == 0 or onenum == 0:
        return 0.0
    length2 = len(ampen2)
    pzero = zeronum / length2
    pone = onenum / length2
    en = np.log2(pzero) * pzero + np.log2(pone) * pone
    return -en


@numba.jit(nopython=True)
def pmi(ampbi, t):
    """
    - Numba JIT 加速
    """
    lengthall = len(ampbi)
    part1 = ampbi[0:t]
    part2 = ampbi[t:]


    zeronum0 = (part1 == 0).sum()
    zeronum1 = (part1 == 1).sum()
    onenum0 = (part2 == 0).sum()
    onenum1 = (part2 == 1).sum()

    pmi_matrix = np.zeros((2, 2))
    npmi_matrix = np.zeros((2, 2))

    total_zeros = zeronum0 + onenum0
    total_ones = zeronum1 + onenum1
    len_part1 = zeronum0 + zeronum1
    len_part2 = onenum0 + onenum1

    if zeronum0 > 0 and total_zeros > 0 and len_part1 > 0:
        denominator = len_part1 * total_zeros
        pmi_val = zeronum0 * lengthall / denominator
        pmi_matrix[0, 0] = (zeronum0 / lengthall) * np.log2(pmi_val)
        npmi_matrix[0, 0] = -np.log2(pmi_val) / np.log2(zeronum0 / lengthall)
    else:
        npmi_matrix[0, 0] = -1.0

    if zeronum1 > 0 and total_ones > 0 and len_part1 > 0:
        denominator = len_part1 * total_ones
        pmi_val = zeronum1 * lengthall / denominator
        pmi_matrix[0, 1] = (zeronum1 / lengthall) * np.log2(pmi_val)
        npmi_matrix[0, 1] = -np.log2(pmi_val) / np.log2(zeronum1 / lengthall)
    else:
        npmi_matrix[0, 1] = -1.0

    if onenum0 > 0 and total_zeros > 0 and len_part2 > 0:
        denominator = len_part2 * total_zeros
        pmi_val = onenum0 * lengthall / denominator
        pmi_matrix[1, 0] = (onenum0 / lengthall) * np.log2(pmi_val)
        npmi_matrix[1, 0] = -np.log2(pmi_val) / np.log2(onenum0 / lengthall)
    else:
        npmi_matrix[1, 0] = -1.0

    if onenum1 > 0 and total_ones > 0 and len_part2 > 0:
        denominator = len_part2 * total_ones
        pmi_val = onenum1 * lengthall / denominator
        pmi_matrix[1, 1] = (onenum1 / lengthall) * np.log2(pmi_val)
        npmi_matrix[1, 1] = -np.log2(pmi_val) / np.log2(onenum1 / lengthall)
    else:
        npmi_matrix[1, 1] = -1.0
    mivalue = pmi_matrix.sum()
    enpmivalue = (npmi_matrix[0, 0] * zeronum0 - npmi_matrix[0, 1] * zeronum1 -
                  npmi_matrix[1, 0] * onenum0 + npmi_matrix[1, 1] * onenum1) / lengthall
    return mivalue, enpmivalue


@numba.jit(nopython=True)
def maxpmi(ampbi, n):

    zeronum = (ampbi == 0).sum()
    onenum = (ampbi == 1).sum()

    if zeronum == 0 or onenum == 0:
        return -1.0, -1.0, np.array([-1], dtype=np.int64)

    if n != -1:
        mi_val, enpmi_val = pmi(ampbi, n)
        return mi_val, enpmi_val, np.array([n], dtype=np.int64)
    else:
        readyindex = np.where(ampbi[1:] - ampbi[:-1] == 1)[0]
        if len(readyindex) == 0:
            return -1.0, -1.0, np.array([-1], dtype=np.int64)
        results = np.zeros((len(readyindex), 2))
        for i in range(len(readyindex)):
            results[i, 0], results[i, 1] = pmi(ampbi, readyindex[i] + 1)
        max_mi_value = np.max(results[:, 0])
        t_num_indices = np.where(results[:, 0] == max_mi_value)[0]

        final_t_num = readyindex[t_num_indices] + 1
        final_mi_value = results[t_num_indices[0], 0]
        final_enpmi_value = results[t_num_indices[0], 1]
        return final_mi_value, final_enpmi_value, final_t_num


@numba.jit(nopython=True)
def calculateGeneral(xsquare, n):
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
    else:  # n % 2 == 0
        m = (n - 2) / 2.0
        prod = 1.0
        for val in range(1, n - 2, 2):
            prod *= val
        return prod * np.sqrt(np.pi / 2.0 / xsquare) * (1.0 / xsquare) ** m
