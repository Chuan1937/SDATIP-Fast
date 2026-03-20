"""Plotting utilities for seismic waveform analysis results."""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_result(wf, state, qualifiedid, name, outputdir):
    """Generate detailed probability plot for analysis results.

    Args:
        wf: Waveform object with processed data.
        state: State object with probability estimates.
        qualifiedid: Index of the solution to plot.
        name: Station name for filename.
        outputdir: Directory to save the plot.
    """
    plt.rcParams.update({
        "font.weight": "bold",
        "axes.labelweight": "bold",
        "font.family": "Times New Roman",
        "font.size": 45,
    })

    fig = plt.figure(figsize=(25, 9))
    timeprob = state.timeprob[qualifiedid]

    b_upthreshold = np.array(state.upthreshold)
    b_downthreshold = np.array(state.downthreshold)
    b_Apeak = np.array([item[0] for item in state.Apeak])

    a_cut = np.array(wf.cut).flatten().astype(int)

    b_upthreshold[-1] = b_upthreshold[-2] * 1.5

    prob1 = timeprob / (b_upthreshold - b_downthreshold)
    alphacoefficient = (0.75 - 0.03) / np.max(prob1)
    alphas = 0.03 + prob1 * alphacoefficient

    colori = np.zeros((state.num, 3))
    colori[b_Apeak > 0] = [1, 0, 0]
    colori[b_Apeak < 0] = [0, 1, 0]

    mask_nonzero = np.abs(b_Apeak) > 0

    downt = np.zeros(state.num)
    upt = np.zeros(state.num)

    cut_indices = a_cut[mask_nonzero]
    tchange = wf.longtimestamp[cut_indices + 1] - wf.longtimestamp[cut_indices]
    achange = wf.denselongdata[cut_indices + 1] - wf.denselongdata[cut_indices]
    achange[achange == 0] = 1e-9

    downt[mask_nonzero] = (
        tchange / achange * (b_downthreshold[mask_nonzero] - np.abs(wf.denselongdata[cut_indices]))
        + wf.longtimestamp[cut_indices]
    )
    upt[mask_nonzero] = (
        tchange / achange * (b_upthreshold[mask_nonzero] - np.abs(wf.denselongdata[cut_indices]))
        + wf.longtimestamp[cut_indices]
    )

    downt[~mask_nonzero] = wf.longtimestamp[-1]
    upt[~mask_nonzero] = wf.longtimestamp[-1] + 0.1 * wf.timestamp[-1]

    ax1 = fig.add_axes([0.4, 0.33, 0.55, 0.6])
    ax1.plot(wf.longtimestamp, wf.denselongdata, linewidth=2.5, color="k", linestyle="-")
    ax1.plot(wf.longtimestamp, abs(wf.denselongdata), linewidth=2.5, color="k", linestyle=":", alpha=0.9)

    ylim_min = -1 * b_upthreshold[-1]
    ax1.vlines(
        downt[mask_nonzero], ylim_min, b_downthreshold[mask_nonzero],
        color="k", linestyle="--", alpha=0.3, linewidth=0.5
    )
    ax1.vlines(
        upt[mask_nonzero], ylim_min, b_upthreshold[mask_nonzero],
        color="k", linestyle="--", alpha=0.3, linewidth=0.5
    )
    ax1.hlines(
        b_downthreshold[mask_nonzero], 0, downt[mask_nonzero],
        color="k", linestyle="--", alpha=0.3, linewidth=0.5
    )
    ax1.hlines(
        b_upthreshold[mask_nonzero], 0, upt[mask_nonzero],
        color="k", linestyle="--", alpha=0.3, linewidth=0.5
    )

    for i in range(state.num):
        ax1.fill_between(
            [0, downt[i], upt[i]],
            [b_downthreshold[i], b_downthreshold[i], b_upthreshold[i]],
            [b_upthreshold[i], b_upthreshold[i], b_upthreshold[i]],
            color=colori[i], alpha=alphas[i]
        )
        ax1.fill_betweenx(
            [-1 * b_upthreshold[-2], b_downthreshold[i], b_upthreshold[i]],
            [downt[i], downt[i], upt[i]],
            [upt[i], upt[i], upt[i]],
            color=colori[i], alpha=alphas[i]
        )

    ax1.set(
        xlim=(0, np.max(wf.densetimestamp) + 0.1 * wf.timestamp[-1]),
        ylim=(-1 * b_upthreshold[-2], b_upthreshold[-1])
    )
    ax1.tick_params(direction="out", size=20)
    ax1.set_yticks(np.array([-1 * b_upthreshold[-2], 0, b_upthreshold[-1]]))
    ax1.set_yticklabels(
        ["%.2f" % (-1 * b_upthreshold[-2]), "%.2f" % 0, "%.2f" % b_upthreshold[-1]],
        fontweight="bold"
    )
    ax1.set_xticks(np.linspace(0, wf.timestamp[-1], 5))
    ax1.set_xticklabels(
        ["%.2f" % (0), "%.2f" % (wf.timestamp[-1] / 4), "%.2f" % (wf.timestamp[-1] / 4 * 2),
         "%.2f" % (wf.timestamp[-1] / 4 * 3), "%.2f" % (wf.timestamp[-1])],
        fontweight="bold"
    )

    for spine in ax1.spines.values():
        spine.set_linewidth(2)

    ax2 = fig.add_axes([
        0.1, 0.93 - 0.6 / (b_upthreshold[-1] + b_upthreshold[-2]) * b_upthreshold[-1],
        0.25, 0.6 / (b_upthreshold[-1] + b_upthreshold[-2]) * b_upthreshold[-1]
    ])
    ax2.invert_xaxis()

    ax2.barh(
        y=b_downthreshold, width=prob1, height=b_upthreshold - b_downthreshold,
        left=0, align="edge", color=colori, alpha=1
    )

    ax2.set(ylim=(0, b_upthreshold[-1]))
    max_prob1 = np.max(prob1)
    ax2.set_xticks(np.linspace(0, max_prob1, 5))
    ax2.set_xticklabels(
        ["%.2f" % (0), "%.2f" % (max_prob1 / 4), "%.2f" % (max_prob1 / 2),
         "%.2f" % (max_prob1 * 3 / 4), "%.2f" % (max_prob1)],
        fontweight="bold"
    )
    ax2.set_yticks(np.linspace(0, b_upthreshold[-1], 5))
    ax2.set_yticklabels(
        ["%.2f" % (0), "%.2f" % (b_upthreshold[-1] / 4), "%.2f" % (b_upthreshold[-1] / 2),
         "%.2f" % (b_upthreshold[-1] * 3 / 4), "%.2f" % (b_upthreshold[-1])],
        fontweight="bold"
    )
    ax2.tick_params(direction="out", size=20)
    ax2.set_ylabel(r"$\mathbf{\varepsilon_{threshold}}$", weight="bold")
    ax2.set_xlabel(r"PDF of $\mathbf{\varepsilon_{threshold}}$", weight="bold")
    for spine in ax2.spines.values():
        spine.set_linewidth(2)

    ax3 = fig.add_axes([0.4, 0.1, 0.55, 0.15], sharex=ax1)

    max_prob1_1 = np.max(prob1) * 1.1
    ax3.vlines(downt, 0, max_prob1_1, color="k", linestyle="-", alpha=0.3, linewidth=0.5)
    ax3.vlines(upt, 0, max_prob1_1, color="k", linestyle="-", alpha=0.3, linewidth=0.5)
    ax3.hlines(prob1, downt, upt, color="k", linestyle="-", alpha=0.3, linewidth=0.5)
    ax3.bar(x=downt, height=prob1, width=upt - downt, align="edge", color=colori, alpha=1)
    ax3.plot(
        [state.arrivalestimate, state.arrivalestimate], [0, max_prob1_1],
        linewidth=2.3, color="k", linestyle=":", alpha=0.9
    )

    ax3.set(ylim=(0, max_prob1_1))
    ax3.set_yticks(np.linspace(0, np.max(prob1), 3))
    ax3.set_yticklabels(
        ["%.2f" % (0), "%.2f" % (np.max(prob1) / 2), "%.2f" % (np.max(prob1))],
        fontweight="bold"
    )
    ax3.set_xticks(np.linspace(0, wf.timestamp[-1], 5))
    ax3.set_xticklabels(
        ["%.2f" % (0), "%.2f" % (wf.timestamp[-1] / 4), "%.2f" % (wf.timestamp[-1] / 2),
         "%.2f" % (wf.timestamp[-1] * 3 / 4), "%.2f" % (wf.timestamp[-1])],
        fontweight="bold"
    )
    ax3.tick_params(direction="out", size=20)
    ax3.set_ylabel("PDF of Time", weight="bold")
    ax3.set_xlabel("Time/s", weight="bold")
    for spine in ax3.spines.values():
        spine.set_linewidth(2)

    ax4 = fig.add_axes([0.1, 0.13, 0.23, 0.05])

    width = [float(state.polarityup), float(state.polarityunknown), float(state.polaritydown)]
    left = [0, width[0], width[0] + width[1]]
    colors = [[1, 0, 0], [0.7, 0.7, 0.7], [0, 1, 0]]
    labels = ["Up", "Unknown", "Down"]
    ax4.barh(y=[1, 1, 1], width=width, height=1, left=left, color=colors)

    ax4.set_xticks([0.5])
    ax4.set_xticklabels(["0.5"], fontweight="bold")
    ax4.text(0, 2, "Up:%.1f%%" % (width[0] * 100))
    ax4.text(0.85, 2, "Down:%.1f%%" % (width[2] * 100))
    ax4.set_yticks([])
    ax4.set(xlim=(0, 1), ylim=(0.5, 1.5))
    ax4.plot([0.5, 0.5], [0.5, 1.5], linewidth=2.5, color="k", linestyle=":", alpha=1)
    for spine in ax4.spines.values():
        spine.set_linewidth(2)
    ax4.legend(
        handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in colors],
        labels=labels, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.5)
    )

    fig.text(0.21, 0.38, "%s" % (name), {"fontweight": "bold", "fontsize": 25}, horizontalalignment="center")
    fig.text(0.24, 0.33, r"$\mathbf{A_{peak}}$" + ": %.3f" % (state.Apeakestimate), {"fontweight": "bold", "fontsize": 15})
    fig.text(0.24, 0.28, r"$\mathbf{\sigma}$" + ": %.3f" % (state.sigmaestimate), {"fontweight": "bold", "fontsize": 15})
    fig.text(0.1, 0.33, "Arrivaltime" + ": %.3f" % (state.arrivalestimate), {"fontweight": "bold", "fontsize": 15})
    fig.text(0.1, 0.28, "Polarity Up" + ": %.3f" % (state.polarityestimation), {"fontweight": "bold", "fontsize": 15})
    fig.text(0.1, 0.23, "Eig value:" + " %s" % (state.bigeig))

    fig.savefig("%s" % (outputdir) + "%s_%d.pdf" % (name, qualifiedid))
    plt.close(fig)


def plot_result_graduate(wf, state, qualifiedid, name, outputdir):
    """Generate graduate-style plot for publication.

    Args:
        wf: Waveform object with processed data.
        state: State object with probability estimates.
        qualifiedid: Index of the solution to plot.
        name: Station name for filename.
        outputdir: Directory to save the plot.
    """
    plt.rcParams.update({
        "font.weight": "normal",
        "axes.labelweight": "normal",
        "font.family": "Times New Roman",
        "font.size": 45,
    })

    fig = plt.figure(figsize=(20, 12))
    timeprob = state.timeprob[qualifiedid]

    b_upthreshold = np.array(state.upthreshold)
    b_downthreshold = np.array(state.downthreshold)
    b_Apeak = np.array([item[0] for item in state.Apeak])

    a_cut = np.array(wf.cut).flatten().astype(int)

    b_upthreshold[-1] = b_upthreshold[-2] * 1.5

    prob1 = timeprob / (b_upthreshold - b_downthreshold)
    alphacoefficient = 0.75 / np.max(prob1)
    alphas = timeprob / (b_upthreshold - b_downthreshold) * alphacoefficient
    alphas_fill = 0.03 + alphas

    colori = np.zeros((state.num, 3))
    colori[b_Apeak > 0] = [1, 0, 0]
    colori[b_Apeak < 0] = [0, 0, 1]

    oritprob = np.zeros(wf.length)

    tprobid = np.floor(wf.longtimestamp[a_cut] / wf.delta).astype(int)
    np.add.at(oritprob, tprobid, timeprob)

    mask_nonzero = np.abs(b_Apeak) > 0

    downt1 = np.zeros(state.num)
    upt1 = np.zeros(state.num)

    downt1[mask_nonzero] = wf.timestamp[tprobid[mask_nonzero]]
    upt1[mask_nonzero] = wf.timestamp[tprobid[mask_nonzero] + 1]
    downt1[~mask_nonzero] = wf.timestamp[-1]
    upt1[~mask_nonzero] = wf.timestamp[-1] + 0.1 * wf.timestamp[-1]

    downt = np.zeros(state.num)
    upt = np.zeros(state.num)

    cut_indices = a_cut[mask_nonzero]
    tchange = wf.longtimestamp[cut_indices + 1] - wf.longtimestamp[cut_indices]
    achange = wf.denselongdata[cut_indices + 1] - wf.denselongdata[cut_indices]
    achange[achange == 0] = 1e-9

    downt[mask_nonzero] = (
        tchange / achange * (b_downthreshold[mask_nonzero] - np.abs(wf.denselongdata[cut_indices]))
        + wf.longtimestamp[cut_indices]
    )
    upt[mask_nonzero] = (
        tchange / achange * (b_upthreshold[mask_nonzero] - np.abs(wf.denselongdata[cut_indices]))
        + wf.longtimestamp[cut_indices]
    )

    downt[~mask_nonzero] = wf.longtimestamp[-1]
    upt[~mask_nonzero] = wf.longtimestamp[-1] + 0.1 * wf.timestamp[-1]

    ax1 = fig.add_axes([0.47, 0.33, 0.45, 0.6])
    ax1.plot(wf.longtimestamp, wf.denselongdata, linewidth=5, color="k", linestyle="-")
    ax1.plot(wf.longtimestamp, abs(wf.denselongdata), linewidth=5, color="k", linestyle=":", alpha=0.9)

    for i in range(state.num):
        ax1.fill_between(
            [0, downt[i], upt[i]],
            [b_downthreshold[i], b_downthreshold[i], b_upthreshold[i]],
            [b_upthreshold[i], b_upthreshold[i], b_upthreshold[i]],
            color=colori[i], alpha=alphas_fill[i]
        )
        ax1.fill_betweenx(
            [-1 * b_upthreshold[-2], b_downthreshold[i], b_upthreshold[i]],
            [downt[i], downt[i], upt[i]],
            [upt[i], upt[i], upt[i]],
            color=colori[i], alpha=alphas[i]
        )

    ax1.set(
        xlim=(0, np.max(wf.densetimestamp) + 0.1 * wf.timestamp[-1]),
        ylim=(-1 * b_upthreshold[-2], b_upthreshold[-1])
    )
    ax1.yaxis.tick_right()
    ax1.tick_params(direction="out", size=20, length=5, width=2)
    ax1.set_yticks(np.array([-1 * b_upthreshold[-2], 0, b_upthreshold[-1]]))
    ax1.set_yticklabels([
        "%.1f" % (-1 * b_upthreshold[-2] / 10000),
        "%.1f" % (0),
        "%.1f" % (b_upthreshold[-1] / 10000)
    ])
    ax1.set_xticks(np.linspace(0, wf.timestamp[-1], 5))
    ax1.set_xticklabels(
        ["%.2f" % (0), "%.2f" % (wf.timestamp[-1] / 4), "%.2f" % (wf.timestamp[-1] / 4 * 2),
         "%.2f" % (wf.timestamp[-1] / 4 * 3), "%.2f" % (wf.timestamp[-1])],
        fontsize=25
    )
    [t.set_color("white") for t in ax1.xaxis.get_ticklabels()]
    for spine in ax1.spines.values():
        spine.set_linewidth(5)

    ax2 = fig.add_axes([
        0.11, 0.93 - 0.6 / (b_upthreshold[-1] + b_upthreshold[-2]) * b_upthreshold[-1],
        0.25, 0.6 / (b_upthreshold[-1] + b_upthreshold[-2]) * b_upthreshold[-1]
    ])
    ax2.invert_xaxis()
    ax2.barh(
        y=b_downthreshold, width=prob1, height=b_upthreshold - b_downthreshold,
        left=0, align="edge", color=colori, alpha=1
    )

    ax2.set(ylim=(-4000, b_upthreshold[-1]))
    max_prob1 = np.max(prob1)
    ax2.set_xticks(np.linspace(0, max_prob1, 3))
    ax2.set_xticklabels(["%d" % (0), "%.2f" % (max_prob1 / 2 * 70), "%.2f" % (max_prob1 * 70)])
    ax2.set_yticks(np.linspace(0, b_upthreshold[-1], 3))
    ax2.set_yticklabels([
        "%.1f" % (0),
        "%.1f" % (b_upthreshold[-1] / 2 / 10000),
        "%.1f" % (b_upthreshold[-1] / 10000)
    ])
    ax2.tick_params(direction="out", size=20, length=5, width=2)
    ax2.set_ylabel(r"$\epsilon$")
    ax2.set_xlabel("PDF")
    for spine in ax2.spines.values():
        spine.set_linewidth(5)

    ax3 = fig.add_axes([0.47, 0.13, 0.45, 0.12])
    ax3.bar(x=downt1, height=oritprob[tprobid], width=upt1 - downt1, align="edge", color=colori, alpha=1)

    max_oritprob = np.max(oritprob)
    ax3.plot(
        [state.arrivalestimate, state.arrivalestimate], [0, max_oritprob * 1.1],
        linewidth=3, color="k", linestyle=":", alpha=0.9
    )
    ax3.yaxis.tick_right()
    ax3.set(
        xlim=(0, np.max(wf.densetimestamp) + 0.1 * wf.timestamp[-1]),
        ylim=(0, max_oritprob * 1.1)
    )
    ax3.set_yticks(np.linspace(0, max_oritprob, 3))
    ax3.set_yticklabels(["%d" % (0), "%d" % (100 * int(max_oritprob / 2)), "%d" % (100 * int(max_oritprob))])
    ax3.set_xticks(np.linspace(0, wf.timestamp[-1], 5))
    ax3.set_xticklabels(
        ["%.2f" % (0), "%.2f" % (wf.timestamp[-1] / 4), "%.2f" % (wf.timestamp[-1] / 4 * 2),
         "%.2f" % (wf.timestamp[-1] / 4 * 3), "%.2f" % (wf.timestamp[-1])],
        fontsize=45
    )
    ax3.tick_params(direction="out", size=20, length=5, width=1)
    ax3.set_ylabel("PDF")
    ax3.set_xlabel("Time (s)")
    for spine in ax3.spines.values():
        spine.set_linewidth(5)

    ax4 = fig.add_axes([0.125, 0.12, 0.225, 0.05])
    width = [float(state.polarityup), float(state.polarityunknown), float(state.polaritydown)]
    left = [0, width[0], width[0] + width[1]]
    colors = [[1, 0, 0], [0.7, 0.7, 0.7], [0, 0, 1]]
    ax4.barh(y=[1, 1, 1], width=width, height=1, left=left, color=colors)

    ax4.set_xticks([0.5])
    ax4.set_xticklabels(["0.5"])
    ax4.text(-0.07, 2, "U: %.1f%%" % (abs(width[0]) * 100), fontsize=45)
    ax4.text(0.7, 2, "D: %.1f%%" % (abs(width[2]) * 100), fontsize=45)
    ax4.text(-0.37, 2, "Pol:", fontsize=45)
    ax4.set_yticks([])
    ax4.set(xlim=(0, 1), ylim=(0.5, 1.5))
    ax4.plot([0.5, 0.5], [0.5, 1.5], linewidth=5, color="k", linestyle=":", alpha=1)
    for spine in ax4.spines.values():
        spine.set_linewidth(5)

    fig.text(0.04, 0.305, f"E.V.: {state.bigeig[0]:.0f}, {state.bigeig[1]:.0f}, {state.bigeig[2]:.2f}, ...", fontsize=45)

    fig.savefig("%s" % (outputdir) + "%s.eps" % (name))
    fig.savefig("%s" % (outputdir) + "%s_%d.pdf" % (name, qualifiedid))
    plt.close(fig)
