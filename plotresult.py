import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plotprob(a, b, qualifiedid, name, outputdir):

    plt.rcParams.update({
        "font.weight": "bold",
        "axes.labelweight": "bold",
        "font.family": "Times New Roman", 
        "font.size": 45
    })

    fig = plt.figure(figsize=(25, 9))
    timeprob = b.timeprob[qualifiedid]


    b_upthreshold = np.array(b.upthreshold)
    b_downthreshold = np.array(b.downthreshold)
    b_Apeak = np.array([item[0] for item in b.Apeak])


    a_cut = np.array(a.cut).flatten().astype(int)

    b_upthreshold[-1] = b_upthreshold[-2] * 1.5


    prob1 = timeprob / (b_upthreshold - b_downthreshold)
    alphacoefficient = (0.75 - 0.03) / np.max(prob1)
    alphas = 0.03 + prob1 * alphacoefficient

 
    colori = np.zeros((b.num, 3))
    colori[b_Apeak > 0] = [1, 0, 0]  
    colori[b_Apeak < 0] = [0, 1, 0]  

    mask_nonzero = np.abs(b_Apeak) > 0

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


    ax1 = fig.add_axes([0.4, 0.33, 0.55, 0.6])
    ax1.plot(a.longtimestamp, a.denselongdata, linewidth=2.5, color='k', linestyle='-')
    ax1.plot(a.longtimestamp, abs(a.denselongdata), linewidth=2.5, color='k', linestyle=':', alpha=0.9)


    ylim_min = -1 * b_upthreshold[-1]
    ax1.vlines(downt[mask_nonzero], ylim_min, b_downthreshold[mask_nonzero], color='k', linestyle='--', alpha=0.3,
               linewidth=0.5)
    ax1.vlines(upt[mask_nonzero], ylim_min, b_upthreshold[mask_nonzero], color='k', linestyle='--', alpha=0.3,
               linewidth=0.5)
    ax1.hlines(b_downthreshold[mask_nonzero], 0, downt[mask_nonzero], color='k', linestyle='--', alpha=0.3,
               linewidth=0.5)
    ax1.hlines(b_upthreshold[mask_nonzero], 0, upt[mask_nonzero], color='k', linestyle='--', alpha=0.3, linewidth=0.5)


    for i in range(b.num):
        ax1.fill_between([0, downt[i], upt[i]], [b_downthreshold[i], b_downthreshold[i], b_upthreshold[i]],
                         [b_upthreshold[i], b_upthreshold[i], b_upthreshold[i]], color=colori[i], alpha=alphas[i])
        ax1.fill_betweenx([-1 * b_upthreshold[-2], b_downthreshold[i], b_upthreshold[i]], [downt[i], downt[i], upt[i]],
                          [upt[i], upt[i], upt[i]], color=colori[i], alpha=alphas[i])

  
    ax1.set(xlim=(0, np.max(a.densetimestamp) + 0.1 * a.timestamp[-1]),
            ylim=(-1 * b_upthreshold[-2], b_upthreshold[-1]))
    ax1.tick_params(direction='out', size=20)
    ax1.set_yticks(np.array([-1 * b_upthreshold[-2], 0, b_upthreshold[-1]]))
    ax1.set_yticklabels(['%.2f' % (-1 * b_upthreshold[-2]), '%.2f' % 0, '%.2f' % b_upthreshold[-1]],
                        **{'fontweight': 'bold'})
    ax1.set_xticks(np.linspace(0, a.timestamp[-1], 5))
    ax1.set_xticklabels(['%.2f' % (0), '%.2f' % (a.timestamp[-1] / 4), '%.2f' % (a.timestamp[-1] / 4 * 2),
                         '%.2f' % (a.timestamp[-1] / 4 * 3), '%.2f' % (a.timestamp[-1])], **{'fontweight': 'bold'})

    for spine in ax1.spines.values(): spine.set_linewidth(2)


    ax2 = fig.add_axes([0.1, 0.93 - 0.6 / (b_upthreshold[-1] + b_upthreshold[-2]) * b_upthreshold[-1], 0.25,
                        0.6 / (b_upthreshold[-1] + b_upthreshold[-2]) * b_upthreshold[-1]])
    ax2.invert_xaxis()

    ax2.barh(y=b_downthreshold, width=prob1, height=b_upthreshold - b_downthreshold, left=0, align='edge', color=colori,
             alpha=1)

 
    ax2.set(ylim=(0, b_upthreshold[-1]))
    max_prob1 = np.max(prob1)
    ax2.set_xticks(np.linspace(0, max_prob1, 5))
    ax2.set_xticklabels(['%.2f' % (0), '%.2f' % (max_prob1 / 4), '%.2f' % (max_prob1 / 2), '%.2f' % (max_prob1 * 3 / 4),
                         '%.2f' % (max_prob1)], **{'fontweight': 'bold'})
    ax2.set_yticks(np.linspace(0, b_upthreshold[-1], 5))
    ax2.set_yticklabels(['%.2f' % (0), '%.2f' % (b_upthreshold[-1] / 4), '%.2f' % (b_upthreshold[-1] / 2),
                         '%.2f' % (b_upthreshold[-1] * 3 / 4), '%.2f' % (b_upthreshold[-1])], **{'fontweight': 'bold'})
    ax2.tick_params(direction='out', size=20)
    ax2.set_ylabel(r'$\mathbf{\varepsilon_{threshold}}$', weight='bold')
    ax2.set_xlabel(r'PDF of $\mathbf{\varepsilon_{threshold}}$', weight='bold')
    for spine in ax2.spines.values(): spine.set_linewidth(2)


    ax3 = fig.add_axes([0.4, 0.1, 0.55, 0.15], sharex=ax1)

    max_prob1_1 = np.max(prob1) * 1.1
    ax3.vlines(downt, 0, max_prob1_1, color='k', linestyle='-', alpha=0.3, linewidth=0.5)
    ax3.vlines(upt, 0, max_prob1_1, color='k', linestyle='-', alpha=0.3, linewidth=0.5)
    ax3.hlines(prob1, downt, upt, color='k', linestyle='-', alpha=0.3, linewidth=0.5)
    ax3.bar(x=downt, height=prob1, width=upt - downt, align='edge', color=colori, alpha=1)
    ax3.plot([b.arrivalestimate, b.arrivalestimate], [0, max_prob1_1], linewidth=2.3, color='k', linestyle=':',
             alpha=0.9)


    ax3.set(ylim=(0, max_prob1_1))
    ax3.set_yticks(np.linspace(0, np.max(prob1), 3))
    ax3.set_yticklabels(['%.2f' % (0), '%.2f' % (np.max(prob1) / 2), '%.2f' % (np.max(prob1))],
                        **{'fontweight': 'bold'})
    ax3.set_xticks(np.linspace(0, a.timestamp[-1], 5))
    ax3.set_xticklabels(['%.2f' % (0), '%.2f' % (a.timestamp[-1] / 4), '%.2f' % (a.timestamp[-1] / 2),
                         '%.2f' % (a.timestamp[-1] * 3 / 4), '%.2f' % (a.timestamp[-1])], **{'fontweight': 'bold'})
    ax3.tick_params(direction='out', size=20)
    ax3.set_ylabel('PDF of Time', weight='bold')
    ax3.set_xlabel('Time/s', weight='bold')
    for spine in ax3.spines.values(): spine.set_linewidth(2)

    ax4 = fig.add_axes([0.1, 0.13, 0.23, 0.05])
    
    width = [float(b.polarityup), float(b.polarityunknown), float(b.polaritydown)]
    left = [0, width[0], width[0] + width[1]]
    colors = [[1, 0, 0], [0.7, 0.7, 0.7], [0, 1, 0]]
    labels = ['Up', 'Unknown', 'Down']
    ax4.barh(y=[1, 1, 1], width=width, height=1, left=left, color=colors)


    ax4.set_xticks([0.5])
    ax4.set_xticklabels(['0.5'], **{'fontweight': 'bold'})
    ax4.text(0, 2, 'Up:%.1f%%' % (width[0] * 100))
    ax4.text(0.85, 2, 'Down:%.1f%%' % (width[2] * 100))
    ax4.set_yticks([])
    ax4.set(xlim=(0, 1), ylim=(0.5, 1.5))
    ax4.plot([0.5, 0.5], [0.5, 1.5], linewidth=2.5, color='k', linestyle=':', alpha=1)
    for spine in ax4.spines.values(): spine.set_linewidth(2)
    ax4.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in colors], labels=labels, ncol=3,
               loc='lower center', bbox_to_anchor=(0.5, 1.5))


    fig.text(0.21, 0.38, '%s' % (name), {'fontweight': 'bold', 'fontsize': 25}, horizontalalignment='center')
    fig.text(0.24, 0.33, r'$\mathbf{A_{peak}}$' + ': %.3f' % (b.Apeakestimate), {'fontweight': 'bold', 'fontsize': 15})
    fig.text(0.24, 0.28, r'$\mathbf{\sigma}$' + ': %.3f' % (b.sigmaestimate), {'fontweight': 'bold', 'fontsize': 15})
    fig.text(0.1, 0.33, 'Arrivaltime' + ': %.3f' % (b.arrivalestimate), {'fontweight': 'bold', 'fontsize': 15})
    fig.text(0.1, 0.28, 'Polarity Up' + ': %.3f' % (b.polarityestimation), {'fontweight': 'bold', 'fontsize': 15})
    fig.text(0.1, 0.23, 'Eig value:' + ' %s' % (b.bigeig))

    # fig.savefig('%s' % (outputdir) + '%s_%d.png' % (name, qualifiedid), dpi=600)
    fig.savefig('%s' % (outputdir) + '%s_%d.pdf' % (name, qualifiedid))
    plt.close(fig)
