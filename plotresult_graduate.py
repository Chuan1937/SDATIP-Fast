import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plotprob(a, b, qualifiedid, name, outputdir):

    plt.rcParams.update({
        "font.weight": "normal",
        "axes.labelweight": "normal",
        "font.family": "Times New Roman",
        "font.size": 45
    })

    fig = plt.figure(figsize=(20, 12))
    timeprob = b.timeprob[qualifiedid]


    b_upthreshold = np.array(b.upthreshold)
    b_downthreshold = np.array(b.downthreshold)
    b_Apeak = np.array([item[0] for item in b.Apeak])


    a_cut = np.array(a.cut).flatten().astype(int)

    b_upthreshold[-1] = b_upthreshold[-2] * 1.5


    prob1 = timeprob / (b_upthreshold - b_downthreshold)
    alphacoefficient = 0.75 / np.max(prob1)
    alphas = timeprob / (b_upthreshold - b_downthreshold) * alphacoefficient
    alphas_fill = 0.03 + alphas

    colori = np.zeros((b.num, 3))
    colori[b_Apeak > 0] = [1, 0, 0]
    colori[b_Apeak < 0] = [0, 0, 1]

    oritprob = np.zeros(a.length)

    tprobid = np.floor(a.longtimestamp[a_cut] / a.delta).astype(int)
    np.add.at(oritprob, tprobid, timeprob)

    mask_nonzero = np.abs(b_Apeak) > 0

    downt1 = np.zeros(b.num)
    upt1 = np.zeros(b.num)

    downt1[mask_nonzero] = a.timestamp[tprobid[mask_nonzero]]
    upt1[mask_nonzero] = a.timestamp[tprobid[mask_nonzero] + 1]
    downt1[~mask_nonzero] = a.timestamp[-1]
    upt1[~mask_nonzero] = a.timestamp[-1] + 0.1 * a.timestamp[-1]

    downt = np.zeros(b.num)
    upt = np.zeros(b.num)


    cut_indices = a_cut[mask_nonzero]
    tchange = a.longtimestamp[cut_indices + 1] - a.longtimestamp[cut_indices]
    achange = a.denselongdata[cut_indices + 1] - a.denselongdata[cut_indices]
    achange[achange == 0] = 1e-9

    downt[mask_nonzero] = tchange / achange * (b_downthreshold[mask_nonzero] - np.abs(a.denselongdata[cut_indices])) + \
                          a.longtimestamp[cut_indices]
    upt[mask_nonzero] = tchange / achange * (b_upthreshold[mask_nonzero] - np.abs(a.denselongdata[cut_indices])) + \
                        a.longtimestamp[cut_indices]

    downt[~mask_nonzero] = a.longtimestamp[-1]
    upt[~mask_nonzero] = a.longtimestamp[-1] + 0.1 * a.timestamp[-1]


    ax1 = fig.add_axes([0.47, 0.33, 0.45, 0.6])
    ax1.plot(a.longtimestamp, a.denselongdata, linewidth=5, color='k', linestyle='-')
    ax1.plot(a.longtimestamp, abs(a.denselongdata), linewidth=5, color='k', linestyle=':', alpha=0.9)

    for i in range(b.num):
        ax1.fill_between([0, downt[i], upt[i]], [b_downthreshold[i], b_downthreshold[i], b_upthreshold[i]],
                         [b_upthreshold[i], b_upthreshold[i], b_upthreshold[i]], color=colori[i], alpha=alphas_fill[i])
        ax1.fill_betweenx([-1 * b_upthreshold[-2], b_downthreshold[i], b_upthreshold[i]], [downt[i], downt[i], upt[i]],
                          [upt[i], upt[i], upt[i]], color=colori[i], alpha=alphas[i])

    ax1.set(xlim=(0, np.max(a.densetimestamp) + 0.1 * a.timestamp[-1]),
            ylim=(-1 * b_upthreshold[-2], b_upthreshold[-1]))
    ax1.yaxis.tick_right()
    ax1.tick_params(direction='out', size=20, length=5, width=2)
    ax1.set_yticks(np.array([-1 * b_upthreshold[-2], 0, b_upthreshold[-1]]))
    ax1.set_yticklabels(['%.1f' % (-1 * b_upthreshold[-2] / 10000), '%.1f' % (0), '%.1f' % (b_upthreshold[-1] / 10000)])
    ax1.set_xticks(np.linspace(0, a.timestamp[-1], 5))
    ax1.set_xticklabels(['%.2f' % (0), '%.2f' % (a.timestamp[-1] / 4), '%.2f' % (a.timestamp[-1] / 4 * 2),
                         '%.2f' % (a.timestamp[-1] / 4 * 3), '%.2f' % (a.timestamp[-1])], fontsize=25)
    [t.set_color('white') for t in ax1.xaxis.get_ticklabels()]
    for spine in ax1.spines.values(): spine.set_linewidth(5)

    ax2 = fig.add_axes([0.11, 0.93 - 0.6 / (b_upthreshold[-1] + b_upthreshold[-2]) * b_upthreshold[-1], 0.25,
                        0.6 / (b_upthreshold[-1] + b_upthreshold[-2]) * b_upthreshold[-1]])
    ax2.invert_xaxis()
    ax2.barh(y=b_downthreshold, width=prob1, height=b_upthreshold - b_downthreshold, left=0, align='edge', color=colori,
             alpha=1)

    ax2.set(ylim=(-4000, b_upthreshold[-1]))
    max_prob1 = np.max(prob1)
    ax2.set_xticks(np.linspace(0, max_prob1, 3))
    ax2.set_xticklabels(['%d' % (0), '%.2f' % (max_prob1 / 2 * 70), '%.2f' % (max_prob1 * 70)])
    ax2.set_yticks(np.linspace(0, b_upthreshold[-1], 3))
    ax2.set_yticklabels(['%.1f' % (0), '%.1f' % (b_upthreshold[-1] / 2 / 10000), '%.1f' % (b_upthreshold[-1] / 10000)])
    ax2.tick_params(direction='out', size=20, length=5, width=2)
    ax2.set_ylabel(r'$\epsilon$')
    ax2.set_xlabel('PDF')
    for spine in ax2.spines.values(): spine.set_linewidth(5)

    ax3 = fig.add_axes([0.47, 0.13, 0.45, 0.12])
    ax3.bar(x=downt1, height=oritprob[tprobid], width=upt1 - downt1, align='edge', color=colori, alpha=1)

    max_oritprob = np.max(oritprob)
    ax3.plot([b.arrivalestimate, b.arrivalestimate], [0, max_oritprob * 1.1], linewidth=3, color='k', linestyle=':',
             alpha=0.9)
    ax3.yaxis.tick_right()
    ax3.set(xlim=(0, np.max(a.densetimestamp) + 0.1 * a.timestamp[-1]), ylim=(0, max_oritprob * 1.1))
    ax3.set_yticks(np.linspace(0, max_oritprob, 3))
    ax3.set_yticklabels(['%d' % (0), '%d' % (100 * int(max_oritprob / 2)), '%d' % (100 * int(max_oritprob))])
    ax3.set_xticks(np.linspace(0, a.timestamp[-1], 5))
    ax3.set_xticklabels(['%.2f' % (0), '%.2f' % (a.timestamp[-1] / 4), '%.2f' % (a.timestamp[-1] / 4 * 2),
                         '%.2f' % (a.timestamp[-1] / 4 * 3), '%.2f' % (a.timestamp[-1])], fontsize=45)
    ax3.tick_params(direction='out', size=20, length=5, width=1)
    ax3.set_ylabel('PDF')
    ax3.set_xlabel('Time (s)')
    for spine in ax3.spines.values(): spine.set_linewidth(5)

    ax4 = fig.add_axes([0.125, 0.12, 0.225, 0.05])
    width = [float(b.polarityup), float(b.polarityunknown), float(b.polaritydown)]
    left = [0, width[0], width[0] + width[1]]
    colors = [[1, 0, 0], [0.7, 0.7, 0.7], [0, 0, 1]]
    ax4.barh(y=[1, 1, 1], width=width, height=1, left=left, color=colors)

    ax4.set_xticks([0.5])
    ax4.set_xticklabels(['0.5'])
    ax4.text(-0.07, 2, 'U: %.1f%%' % (abs(width[0]) * 100), fontsize=45)
    ax4.text(0.7, 2, 'D: %.1f%%' % (abs(width[2]) * 100), fontsize=45)
    ax4.text(-0.37, 2, 'Pol:', fontsize=45)
    ax4.set_yticks([])
    ax4.set(xlim=(0, 1), ylim=(0.5, 1.5))
    ax4.plot([0.5, 0.5], [0.5, 1.5], linewidth=5, color='k', linestyle=':', alpha=1)
    for spine in ax4.spines.values(): spine.set_linewidth(5)

    fig.text(0.04, 0.305, f'E.V.: {b.bigeig[0]:.0f}, {b.bigeig[1]:.0f}, {b.bigeig[2]:.2f}, ...', fontsize=45)

 
    fig.savefig('%s' % (outputdir) + '%s.eps' % (name))
    fig.savefig('%s' % (outputdir) + '%s_%d.pdf' % (name, qualifiedid))
    plt.close(fig)
