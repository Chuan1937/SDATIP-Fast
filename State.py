"""State machine for seismic waveform polarity analysis.

This module implements a Markov chain-based state model for computing
arrival time and polarity probabilities of seismic waveforms.
"""

import numba
import numpy as np
from scipy.special import erfc, erf


SQRT2 = np.sqrt(2.0)
SQRT_PI_INV = 1.0 / np.sqrt(np.pi)


@numba.njit(fastmath=True, cache=True)
def erfc_numba(x):
    """Numba-compatible complementary error function approximation.

    Uses Abramowitz & Stegun formula 7.1.26 with max error < 1.5e-7.
    """
    if x < 0:
        return 2.0 - erfc_numba(-x)
    if x > 6.0:
        return SQRT_PI_INV * np.exp(-x * x) / x
    p = 0.3275911
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)
    return 1.0 - y


@numba.njit(fastmath=True, cache=True)
def erf_numba(x):
    """Numba-compatible error function."""
    return 1.0 - erfc_numba(x)


@numba.njit(parallel=True, fastmath=True, cache=True)
def _compute_matrix_single(downthreshold_arr, upthreshold_arr, bestsigma_single, chances_single):
    """Compute transition probabilities for single-state entries in parallel.

    Each row computes: P = exp(-chance*p2) - exp(-chance*p1)
    where p1, p2 are complementary error functions.
    """
    n_states = len(bestsigma_single)
    n_thresholds = len(downthreshold_arr)
    result = np.zeros((n_states, n_thresholds))

    for i in numba.prange(n_states):
        sigma = bestsigma_single[i]
        chance = chances_single[i]
        for j in range(n_thresholds):
            p1 = erfc_numba(downthreshold_arr[j] / (SQRT2 * sigma))
            p2 = erfc_numba(upthreshold_arr[j] / (SQRT2 * sigma))
            result[i, j] = np.exp(-chance * p2) - np.exp(-chance * p1)

    return result


@numba.njit(fastmath=True, cache=True)
def _compute_matrix_combo(downthreshold_arr, upthreshold_arr, bestsigma_k, chance, combonum):
    """Compute transition probabilities for combo-state entries.

    Averages probabilities across multiple sigma values.
    """
    n_thresholds = len(downthreshold_arr)
    n_sigmas = len(bestsigma_k)
    result = np.zeros(n_thresholds)

    for j in range(n_thresholds):
        acc = 0.0
        for k in range(n_sigmas):
            p1 = erfc_numba(downthreshold_arr[j] / (SQRT2 * bestsigma_k[k]))
            p2 = erfc_numba(upthreshold_arr[j] / (SQRT2 * bestsigma_k[k]))
            acc += np.exp(-chance * p2) - np.exp(-chance * p1)
        result[j] = acc / combonum

    return result


@numba.jit(nopython=True, fastmath=True, cache=True)
def _find_best_combination(alpha, leftrange, rightrange, step, k0s_center_offset, k0s_points):
    """Find optimal coefficient combination for two eigenvector case.

    Searches for k0p, k0s that minimize orthogonality between
    two probability distributions derived from eigenvectors.
    """
    num_k = 1 + int(round((rightrange - leftrange) / step))
    k_vec = np.linspace(leftrange, rightrange, num_k)

    bestfit = np.inf
    best_k0p = 0.0
    best_k0s = 0.0

    dot_00 = np.dot(alpha[0], alpha[0])
    dot_01 = np.dot(alpha[0], alpha[1])
    dot_11 = np.dot(alpha[1], alpha[1])

    for i in range(len(k_vec)):
        k0p = k_vec[i]
        k1p = 1.0 - k0p
        probp = k0p * alpha[0] + k1p * alpha[1]

        k0sco = k0p * dot_00 + k1p * dot_01
        k1sco = k1p * dot_11 + k0p * dot_01
        denominator = k0sco - k1sco
        k0s_center = -k1sco / denominator if denominator != 0 else 0.0

        k0srange = np.linspace(k0s_center - k0s_center_offset, k0s_center + k0s_center_offset, k0s_points)

        min_ortho_for_k = np.inf
        best_k0s_for_k = 0.0

        for k0s_val in k0srange:
            k1s_val = 1.0 - k0s_val
            probs = k0s_val * alpha[0] + k1s_val * alpha[1]
            ortho = np.dot(np.abs(probs), np.abs(probp))

            if ortho < min_ortho_for_k:
                min_ortho_for_k = ortho
                best_k0s_for_k = k0s_val

        if min_ortho_for_k < bestfit:
            bestfit = min_ortho_for_k
            best_k0p = k0p
            best_k0s = best_k0s_for_k

    return best_k0p, best_k0s


class State:
    """Markov state model for seismic waveform analysis.

    Each state represents a noise threshold band with associated
    probability distribution for arrival time and polarity estimation.
    """

    def __init__(self, name):
        self.name = name
        self.num = 0
        self.downthreshold = []
        self.upthreshold = []
        self.sample = []
        self.chances = []
        self.samplength = []
        self.xsquare = []
        self.pmi = []
        self.Apeak = []
        self.combo = []
        self.arrivaltimestamp = []
        self.multiplesolution = 1

    def addstate(self, downthreshold, upthreshold, noisesample, chances, pmi, Apeak, arrivaltime):
        """Add a single noise state to the model."""
        noisesample_arr = np.array(noisesample)
        self.num += 1
        self.combo.append(1)
        self.downthreshold.append(downthreshold)
        self.upthreshold.append(upthreshold)
        self.sample.append(noisesample_arr)
        self.chances.append(chances)
        self.samplength.append(len(noisesample_arr))
        self.xsquare.append(np.sum(noisesample_arr**2))
        self.pmi.append(pmi)
        self.Apeak.append(Apeak)
        self.arrivaltimestamp.append(arrivaltime)

    def addcombo(self, combonum, downthreshold, upthreshold, noisesample, chances, pmi, Apeak, arrivaltime):
        """Add a combined noise state (multiple samples) to the model."""
        self.num += 1
        self.combo.append(combonum)
        self.downthreshold.append(downthreshold)
        self.upthreshold.append(upthreshold)
        self.sample.append(noisesample)
        self.chances.append(chances)

        samplengths = [len(s) for s in noisesample]
        xsquares = [np.sum(np.array(s) ** 2) for s in noisesample]

        self.samplength.append(samplengths)
        self.xsquare.append(xsquares)
        self.pmi.append(pmi)
        self.Apeak.append(Apeak)
        self.arrivaltimestamp.append(arrivaltime)

    def markovmatrix(self):
        """Build and analyze the Markov transition matrix.

        Computes transition probabilities between threshold states,
        then extracts dominant eigenvectors for time probability estimation.
        """
        self.num = len(self.combo)
        self.matrix = np.zeros((self.num, self.num))

        downthreshold_arr = np.array(self.downthreshold, dtype=np.float64)
        upthreshold_arr = np.array(self.upthreshold, dtype=np.float64)

        is_single_state = np.array(self.combo) == 1
        single_indices = np.where(is_single_state)[0]

        if len(single_indices) > 0:
            n_single = len(single_indices)
            bestsigma_single = np.zeros(n_single)
            for idx, i in enumerate(single_indices):
                if self.samplength[i] > 0:
                    sigma = np.sqrt(self.xsquare[i] / self.samplength[i])
                    bestsigma_single[idx] = sigma if sigma > 0 else 1e-9
                else:
                    bestsigma_single[idx] = 1e-9

            chances_single = np.array([self.chances[i][0] for i in single_indices])

            p12_single = _compute_matrix_single(
                downthreshold_arr, upthreshold_arr, bestsigma_single, chances_single
            )
            self.matrix[single_indices, :] = p12_single

        combo_indices = np.where(~is_single_state)[0]

        for i in combo_indices:
            if (i > 0 and self.combo[i] == self.combo[i - 1] and
                    self.samplength[i] == self.samplength[i - 1] and
                    np.array_equal(self.sample[i], self.sample[i - 1])):
                self.matrix[i] = self.matrix[i - 1]
                continue

            combonum = self.combo[i]
            xsquare_k = np.array(self.xsquare[i], dtype=np.float64)
            samplength_k = np.array(self.samplength[i], dtype=np.float64)

            n_sigmas = len(samplength_k)
            bestsigma_k = np.full(n_sigmas, 1e-9, dtype=np.float64)
            for k in range(n_sigmas):
                if samplength_k[k] > 0:
                    sigma = np.sqrt(xsquare_k[k] / samplength_k[k])
                    bestsigma_k[k] = sigma if sigma > 0 else 1e-9

            chance = float(self.chances[i][0])
            p12 = _compute_matrix_combo(
                downthreshold_arr, upthreshold_arr, bestsigma_k, chance, combonum
            )
            self.matrix[i] = p12

        row_sums = self.matrix.sum(axis=1, keepdims=True)
        self.matrix = np.divide(self.matrix, row_sums, out=np.zeros_like(self.matrix), where=row_sums != 0)

        eigenvalue, featurevector = np.linalg.eig(self.matrix.T)
        real_eigenvalue = np.real(eigenvalue)

        self.bigeig = np.around(np.sort(real_eigenvalue)[-5:][::-1], 3)

        eigindex = np.where(real_eigenvalue > 0.98)[0]
        self.eigvalue = eigenvalue[eigindex]

        num_eig = len(eigindex)
        if num_eig == 0:
            self.timeprob = []
            return self.timeprob, 0

        if num_eig == 1:
            eig = eigindex[0]
            vec = np.real(featurevector[:, eig])
            vec_sum = np.sum(vec)
            timeprob = vec / vec_sum if vec_sum != 0 else np.zeros_like(vec)
            self.timeprob = [np.abs(timeprob)]
            return self.timeprob, 1

        if num_eig == 2:
            alpha = np.zeros((2, self.num))
            for i in range(2):
                eig = eigindex[i]
                timeprob_abs = np.abs(np.real(featurevector[:, eig]))
                alpha_sum = np.sum(timeprob_abs)
                alpha[i] = timeprob_abs / alpha_sum if alpha_sum != 0 else np.zeros_like(timeprob_abs)

            k0p, k0s = _find_best_combination(alpha, -50.0, 50.0, 0.001, 1.0, 401)

            k1p = 1.0 - k0p
            k1s = 1.0 - k0s

            probp = k0p * alpha[0] + k1p * alpha[1]
            probs = k0s * alpha[0] + k1s * alpha[1]

            self.timeprob = [np.abs(probp), np.abs(probs)]
            return self.timeprob, 2

        self.timeprob = []
        for i in range(num_eig):
            eig = eigindex[i]
            timeprob_abs = np.abs(np.real(featurevector[:, eig]))
            timeprob_sum = np.sum(timeprob_abs)
            timeprob = timeprob_abs / timeprob_sum if timeprob_sum != 0 else np.zeros_like(timeprob_abs)
            self.timeprob.append(timeprob)
        return self.timeprob, num_eig

    def ampprobcalculate(self):
        """Calculate amplitude probability for upward polarity at each state."""
        print("\n")
        print("calculating ampprob")
        self.ampprob_up = []
        self.bestsigma = []
        sqrt2 = np.sqrt(2)

        for i in range(self.num):
            if self.combo[i] == 1:
                bestsigma = np.sqrt(self.xsquare[i] / self.samplength[i])
                self.bestsigma.append(bestsigma)
                p = 0.5 + 0.5 * erf(self.Apeak[i][0] / (sqrt2 * bestsigma))
                self.ampprob_up.append(p)
            else:
                xsquare_k = np.array(self.xsquare[i])
                samplength_k = np.array(self.samplength[i])
                apeak_k = np.array(self.Apeak[i])
                sigma_k = np.sqrt(xsquare_k / samplength_k)
                p_k = 0.5 + 0.5 * erf(apeak_k / (sqrt2 * sigma_k))
                self.ampprob_up.append(np.mean(p_k))
                self.bestsigma.append(sigma_k.tolist())
        self.ampprob_up = np.clip(self.ampprob_up, a_min=None, a_max=1.0).tolist()
        return np.array(self.ampprob_up)

    def estimation(self, qualifiedid):
        """Estimate arrival time and polarity from the qualified solution."""
        timeprob = np.array(self.timeprob[qualifiedid])
        ampprob_up_arr = np.array(self.ampprob_up)
        self.polarityestimation = np.sum(timeprob * ampprob_up_arr)

        is_unknown = np.array([np.sum(row) == 0 for row in self.Apeak])
        unknownindex = np.where(is_unknown)[0]
        knownindex = np.where(~is_unknown)[0]

        self.polarityunknown = np.sum(timeprob[unknownindex])
        self.polarityup = np.sum(timeprob[knownindex] * ampprob_up_arr[knownindex])
        self.polaritydown = 1 - self.polarityup - self.polarityunknown

        apeakestimate = np.array([np.mean(row) for row in self.Apeak])
        arrivalestimate = np.array([np.mean(row) for row in self.arrivaltimestamp])
        sigmaestimate = np.array([np.mean(row) if isinstance(row, list) else row for row in self.bestsigma])

        self.Apeakestimate = np.sum(timeprob * apeakestimate)
        self.arrivalestimate = np.sum(timeprob * arrivalestimate)
        self.sigmaestimate = np.sum(timeprob * sigmaestimate)

    def getstateinform(self, stateid):
        """Get state information by ID."""
        if stateid >= self.num:
            print("wrong stateid")
            return -1, -1, -1, -1, -1
        return (self.combo[stateid], self.downthreshold[stateid],
                self.upthreshold[stateid], self.sample[stateid],
                self.pmi[stateid], self.Apeak[stateid])

    def getstateprob(self, qualifiedid, stateid):
        """Get state probability by qualified ID and state ID."""
        if stateid >= self.num:
            print("wrong stateid")
            return -1, -1
        return self.timeprob[qualifiedid][stateid], self.ampprob_up[stateid]
