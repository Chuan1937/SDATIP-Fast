import numba
import numpy as np
from scipy.special import erfc, erf



@numba.jit(nopython=True, fastmath=True, cache=True)
def _find_best_combination(alpha, leftrange, rightrange, step, k0s_center_offset, k0s_points):

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


class State():
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
        noisesample_arr = np.array(noisesample)
        self.num += 1
        self.combo.append(1)
        self.downthreshold.append(downthreshold)
        self.upthreshold.append(upthreshold)
        self.sample.append(noisesample_arr)
        self.chances.append(chances)
        self.samplength.append(len(noisesample_arr))
        self.xsquare.append(np.sum(noisesample_arr ** 2))
        self.pmi.append(pmi)
        self.Apeak.append(Apeak)
        self.arrivaltimestamp.append(arrivaltime)

    def addcombo(self, combonum, downthreshold, upthreshold, noisesample, chances, pmi, Apeak, arrivaltime):
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

    # def markovmatrix(self):
    #     
    #     self.num = len(self.combo)
    #     self.matrix = np.zeros([self.num, self.num])
    #
    #     
    #     downthreshold_arr = np.array(self.downthreshold)
    #     upthreshold_arr = np.array(self.upthreshold)
    #     SQRT2 = np.sqrt(2)
    #
    #     
    #     for i in range(self.num):
    #        
    #         if (i > 0 and self.combo[i] == 1 and
    #                 self.samplength[i] == self.samplength[i - 1] and
    #                 np.array_equal(self.sample[i], self.sample[i - 1])):
    #             self.matrix[i] = self.matrix[i - 1]
    #             continue
    #
    #         p12 = np.zeros(self.num)
    #       
    #         if self.combo[i] == 1:
    #             bestsigma = np.sqrt(self.xsquare[i] / self.samplength[i])
    #             if bestsigma == 0: bestsigma = 1e-9  # 避免除以零
    #
    #             p1 = erfc(downthreshold_arr / (SQRT2 * bestsigma))
    #             p2 = erfc(upthreshold_arr / (SQRT2 * bestsigma))
    #
    #             p12 = np.exp(-self.chances[i][0] * p2) - np.exp(-self.chances[i][0] * p1)
    #   
    #         else:
    #             combonum = self.combo[i]
    #             xsquare_k = np.array(self.xsquare[i])
    #             samplength_k = np.array(self.samplength[i])
    #
    #            
    #             samplength_k[samplength_k == 0] = 1
    #             bestsigma_k = np.sqrt(xsquare_k / samplength_k)
    #             bestsigma_k[bestsigma_k == 0] = 1e-9  # 避免除以零
    #
    #            
    #             p1_matrix = erfc(downthreshold_arr[:, np.newaxis] / (SQRT2 * bestsigma_k[np.newaxis, :]))
    #             p2_matrix = erfc(upthreshold_arr[:, np.newaxis] / (SQRT2 * bestsigma_k[np.newaxis, :]))
    #
    #             p12_matrix = np.exp(-self.chances[i][0] * p2_matrix) - np.exp(-self.chances[i][0] * p1_matrix)
    #             p12 = np.sum(p12_matrix, axis=1) / combonum
    #
    #         row_sum = np.sum(p12)
    #         self.matrix[i] = p12 / row_sum if row_sum != 0 else 0
    #
    #    
    #     eigenvalue, featurevector = np.linalg.eig(self.matrix.T)
    #
    #  
    #     bigeigvalue = np.sort(np.real(eigenvalue))[-5:]
    #     self.bigeig = np.around(bigeigvalue[::-1], 3)
    #
    #     eigindex = np.where(np.real(eigenvalue) > 0.98)[0]
    #     self.eigvalue = eigenvalue[eigindex]
    #
    #   
    #     if len(eigindex) == 1:
    #         eig = eigindex[0]
    #         timeprob_sum = np.sum(featurevector[:, eig])
    #         timeprob = np.real(featurevector[:, eig] / timeprob_sum) if timeprob_sum != 0 else np.zeros_like(
    #             featurevector[:, eig])
    #         self.timeprob = [np.abs(timeprob.flatten())]
    #         return self.timeprob, 1
    #
    #     if len(eigindex) == 2:
    #         alpha = np.zeros([2, self.num])
    #         for i in range(2):
    #             eig = eigindex[i]
    #             timeprob_abs = np.real(np.abs(featurevector[:, eig]))
    #             alpha_sum = np.sum(timeprob_abs)
    #             alpha[i] = timeprob_abs / alpha_sum if alpha_sum != 0 else np.zeros_like(timeprob_abs)
    #
    #         leftrange, rightrange, step = -50, 50, 0.001
    #         k_vec = np.linspace(leftrange, rightrange, 1 + int(np.round((rightrange - leftrange) / step)))
    #
    #         bestfit = np.zeros(len(k_vec))
    #         bestk0s_arr = np.zeros(len(k_vec))
    #
    #         dot_00 = np.dot(alpha[0], alpha[0])
    #         dot_01 = np.dot(alpha[0], alpha[1])
    #         dot_11 = np.dot(alpha[1], alpha[1])
    #
    #         for i in range(len(k_vec)):
    #             k0p = k_vec[i]
    #             k1p = 1 - k0p
    #             probp = k0p * alpha[0] + k1p * alpha[1]
    #
    #             k0sco = k0p * dot_00 + k1p * dot_01
    #             k1sco = k1p * dot_11 + k0p * dot_01
    #
    #             denominator = k0sco - k1sco
    #             k0s_center = -1 * k1sco / denominator if denominator != 0 else 0
    #
    #             k0srange = np.linspace(k0s_center - 1, k0s_center + 1, 401)
    #             k1srange = 1 - k0srange
    #
    #             probs_matrix = k0srange[:, np.newaxis] * alpha[0] + k1srange[:, np.newaxis] * alpha[1]
    #             ortho_vec = np.dot(np.abs(probs_matrix), np.abs(probp))
    #
    #             min_ortho_idx = np.argmin(ortho_vec)
    #             bestfit[i] = ortho_vec[min_ortho_idx]
    #             bestk0s_arr[i] = k0srange[min_ortho_idx]
    #
    #         bestpair_idx = np.argmin(bestfit)
    #         k0p = k_vec[bestpair_idx]
    #         k1p = 1 - k0p
    #         k0s = bestk0s_arr[bestpair_idx]
    #         k1s = 1 - k0s
    #
    #         probp = k0p * alpha[0] + k1p * alpha[1]
    #         probs = k0s * alpha[0] + k1s * alpha[1]
    #
    #         self.timeprob = [np.abs(probp), np.abs(probs)]
    #         return self.timeprob, 2
    #
    #     if len(eigindex) >= 3:
    #         self.timeprob = []
    #         for i in range(len(eigindex)):
    #             eig = eigindex[i]
    #             timeprob_abs = np.real(np.abs(featurevector[:, eig]))
    #             timeprob_sum = np.sum(timeprob_abs)
    #             timeprob = timeprob_abs / timeprob_sum if timeprob_sum != 0 else np.zeros_like(timeprob_abs)
    #             self.timeprob.append(abs(timeprob.flatten()))
    #         return self.timeprob, len(eigindex)
    def markovmatrix(self):
       
        self.num = len(self.combo)
        self.matrix = np.zeros((self.num, self.num))

     
        downthreshold_arr = np.array(self.downthreshold)
        upthreshold_arr = np.array(self.upthreshold)
        SQRT2 = np.sqrt(2)

       
        is_single_state = np.array(self.combo) == 1
        single_indices = np.where(is_single_state)[0]

        if len(single_indices) > 0:
          
            bestsigma_single = np.array(
                [np.sqrt(self.xsquare[i] / self.samplength[i]) if self.samplength[i] > 0 else 1e-9 for i in
                 single_indices])
            bestsigma_single[bestsigma_single == 0] = 1e-9

    
            p1 = erfc(downthreshold_arr[:, np.newaxis] / (SQRT2 * bestsigma_single))
            p2 = erfc(upthreshold_arr[:, np.newaxis] / (SQRT2 * bestsigma_single))

            chances_single = np.array([self.chances[i][0] for i in single_indices])

   
            p12_single = np.exp(-chances_single * p2) - np.exp(-chances_single * p1)

          
            self.matrix[single_indices, :] = p12_single.T

    
        combo_indices = np.where(~is_single_state)[0]

        for i in combo_indices:
            
            if (i > 0 and self.combo[i] == self.combo[i - 1] and
                    self.samplength[i] == self.samplength[i - 1] and
                    np.array_equal(self.sample[i], self.sample[i - 1])):
                self.matrix[i] = self.matrix[i - 1]
                continue

            combonum = self.combo[i]
            xsquare_k = np.array(self.xsquare[i])
            samplength_k = np.array(self.samplength[i])

            valid_len_mask = samplength_k > 0
            bestsigma_k = np.full_like(samplength_k, 1e-9, dtype=np.float64)
            bestsigma_k[valid_len_mask] = np.sqrt(xsquare_k[valid_len_mask] / samplength_k[valid_len_mask])
            bestsigma_k[bestsigma_k == 0] = 1e-9

   
            p1_matrix = erfc(downthreshold_arr[:, np.newaxis] / (SQRT2 * bestsigma_k))
            p2_matrix = erfc(upthreshold_arr[:, np.newaxis] / (SQRT2 * bestsigma_k))
            p12_matrix = np.exp(-self.chances[i][0] * p2_matrix) - np.exp(-self.chances[i][0] * p1_matrix)

            p12 = np.sum(p12_matrix, axis=1) / combonum
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

        if num_eig >= 3:
            self.timeprob = []
            for i in range(num_eig):
                eig = eigindex[i]
                timeprob_abs = np.abs(np.real(featurevector[:, eig]))
                timeprob_sum = np.sum(timeprob_abs)
                timeprob = timeprob_abs / timeprob_sum if timeprob_sum != 0 else np.zeros_like(timeprob_abs)
                self.timeprob.append(timeprob)
            return self.timeprob, num_eig

    def ampprobcalculate(self):
        print('\n')
        print('calculating ampprob')
        self.ampprob_up = []
        self.bestsigma = []
        SQRT2 = np.sqrt(2)

        for i in range(self.num):
            if self.combo[i] == 1:
                bestsigma = np.sqrt(self.xsquare[i] / self.samplength[i])
                self.bestsigma.append(bestsigma)
                p = 0.5 + 0.5 * erf(self.Apeak[i][0] / (SQRT2 * bestsigma))
                self.ampprob_up.append(p)
            else:
                xsquare_k = np.array(self.xsquare[i])
                samplength_k = np.array(self.samplength[i])
                Apeak_k = np.array(self.Apeak[i])
                sigma_k = np.sqrt(xsquare_k / samplength_k)
                p_k = 0.5 + 0.5 * erf(Apeak_k / (SQRT2 * sigma_k))
                self.ampprob_up.append(np.mean(p_k))
                self.bestsigma.append(sigma_k.tolist())
        self.ampprob_up = np.clip(self.ampprob_up, a_min=None, a_max=1.0).tolist()
        return np.array(self.ampprob_up)

    def estimation(self, qualifiedid):

        timeprob = np.array(self.timeprob[qualifiedid])
        ampprob_up_arr = np.array(self.ampprob_up)
        self.polarityestimation = np.sum(timeprob * ampprob_up_arr)


        is_unknown = np.array([np.sum(row) == 0 for row in self.Apeak])
        unknownindex = np.where(is_unknown)[0]
        knownindex = np.where(~is_unknown)[0]

        self.polarityunknown = np.sum(timeprob[unknownindex])
        self.polarityup = np.sum(timeprob[knownindex] * ampprob_up_arr[knownindex])
        self.polaritydown = 1 - self.polarityup - self.polarityunknown


        Apeakestimate = np.array([np.mean(row) for row in self.Apeak])
        arrivalestimate = np.array([np.mean(row) for row in self.arrivaltimestamp])
        sigmaestimate = np.array([np.mean(row) if isinstance(row, list) else row for row in self.bestsigma])

        self.Apeakestimate = np.sum(timeprob * Apeakestimate)
        self.arrivalestimate = np.sum(timeprob * arrivalestimate)
        self.sigmaestimate = np.sum(timeprob * sigmaestimate)

    def timeprobinterpolate(self):
        print('coming soon')

    def getstateinform(self, stateid):
        if (stateid >= self.num):
            print('wrong stateid')
            return -1, -1, -1, -1, -1
        else:
            return self.combo[stateid], self.downthreshold[stateid], self.upthreshold[stateid], self.sample[stateid], self.pmi[stateid], self.Apeak[stateid]

    def getstateprob(self, qualifiedid, stateid):
        if (stateid >= self.num):
            print('wrong stateid')
            return -1, -1
        else:
            return self.timeprob[qualifiedid][stateid], self.ampprob_up[stateid]

