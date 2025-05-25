# %%
import numpy as np
import utils_fig as fig_utils
import matplotlib.pyplot as plt
import matplotlib as mpl
import collections
import pickle
import os
import scipy.signal

"""
scp euler:/cluster/work/sachan/vilem/subset2evaluate/computed/16-metric_quality/d*_m*.pkl computed/16-metric_quality/
"""

# load all available
data = []
for di in range(9):
    for mi in range(45):
        fname = f"../computed/16-metric_quality/d{di}_m{mi}.pkl"
        if os.path.exists(fname):
            with open(fname, "rb") as f:
                data.append(pickle.load(f))

data_new = collections.defaultdict(list)
for line in data:
    for method in ["metric_avg", "metric_var", "metric_cons", "diversity", "pyirt_diffdisc"]:
        data_new[method].append({
            "metric": line["metric"],
            "correlation": line["correlation"],
            "spa": line["spa"][method],
        })

# %%
fig_utils.matplotlib_default()
bins = [0.15, 0.25, 0.35, 0.45, 1.0]
bins_label = [0.1, 0.2, 0.3, 0.4, 0.5]

plt.figure(figsize=(4, 2.5))
for method in ["metric_avg", "metric_var", "metric_cons", "diversity", "pyirt_diffdisc"]:
    data_local = sorted(data_new[method], key=lambda x: x["correlation"])
    data_i = np.digitize([v["correlation"] for v in data_local], bins=bins)
    data_y = collections.defaultdict(list)
    for i, v in zip(data_i, data_local):
        data_y[bins[i]].append(v["spa"])
    data_y = [
        np.average(data_y[bins[i]])
        for i in range(len(bins))
    ]
    plt.plot(
        bins_label,
        scipy.signal.savgol_filter(data_y, 2, 1),
        label=method,
        linewidth=1.5,
    )
plt.axhline(
    y=1,
    color="black",
    label="Random",
    linewidth=1.5,
)
plt.gca().spines[['top', 'right']].set_visible(False)

# show y-axis as percentage
plt.gca().yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1, decimals=0))

plt.ylim(0.7, 1.35)

plt.ylabel("Size needed to match\nsoft pairwise accuracy", labelpad=0)

plt.xticks([0.1, 0.2,  0.3, 0.4, 0.5])
plt.xlabel("Metric correlation with human scores")
plt.gca().tick_params(axis='y', which='major', pad=-1)
# remove tiny tick lines
plt.gca().tick_params(left=False)
# axs[1].yticks(axs[0].get_yticks())
# axs[1].set_yticklabels([])
# axs[1].tick_params(left=True)

# avg metrics correlations

metrics_avg = collections.defaultdict(list)
for line in data_new["metric_avg"]:
    metrics_avg[line["metric"]].append(line["correlation"])

metrics_avg = {
    metric: np.average(corrs)
    for metric, corrs in metrics_avg.items()
}

metrics_avg_flat = sorted([
    (metric, corrs)
    for metric, corrs in metrics_avg.items()
], key=lambda x: x[1], reverse=True)
for metric, corr in metrics_avg_flat:
    print(f"{metric}: {corr:.3f}")


METRIC_NAMES = {
    "BLEU": "BLEU",
    "chrF": "ChrF",
    "BLEURT-20": "BLEURT",
    "BERTscore": "BERTscore",
    "prismRef": "Prism",
    "MetricX-23-c": "MetricX",
    "XCOMET-Ensemble": "XCOMET-Ensemble",
    "COMET": "COMET",
    "GEMBA-MQM": "GEMBA",
}

for item in [
    dict(metric="BLEU", line_yy=(0.7, 0.78), text_xy=(-0.0, 0.02)),
    dict(metric="chrF", line_yy=(0.7, 0.78), text_xy=(-0.0, 0.02)),
    dict(metric="BERTscore", line_yy=(0.7, 0.78), text_xy=(-0.00, 0.06)),
    dict(metric="MetricX-23-c", line_yy=(0.7, 0.78), text_xy=(-0.00, 0.02)),
    dict(metric="GEMBA-MQM", line_yy=(0.7, 0.78), text_xy=(0.06, 0.02)),
]:
    metric = item["metric"]
    plt.vlines(
        ymin=item["line_yy"][0], ymax=item["line_yy"][1],
        x=metrics_avg[metric],
        color="gray",
        linestyle="--",
        linewidth=0.5,
    )
    plt.text(
        x=item["text_xy"][0]+metrics_avg[metric],
        y=item["text_xy"][1]+item["line_yy"][0],
        s=METRIC_NAMES[metric],
        fontsize=8,
        va="center",
        ha="right",
    )


# separator between subplots
plt.tight_layout(pad=0)
plt.subplots_adjust(wspace=0.2)
plt.savefig("../figures_pdf/16-metric_quality_performance.pdf")
plt.show()