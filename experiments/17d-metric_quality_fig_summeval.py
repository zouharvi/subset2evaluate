# %%
import numpy as np
import utils_fig as fig_utils
import matplotlib.pyplot as plt
import matplotlib as mpl
import collections
import pickle
import os
import scipy.signal

# scp euler:/cluster/work/sachan/vilem/subset2evaluate/computed/17-metric_quality/d*_m*.pkl computed/17-metric_quality/

# load all available
data = []
for di in range(5):
    for mi in range(119):
        fname = f"../computed/17-metric_quality/d{di}_m{mi}.pkl"
        if os.path.exists(fname):
            with open(fname, "rb") as f:
                data.append(pickle.load(f))

data_new = collections.defaultdict(list)
for line in data:
    for method in ["metric_avg", "metric_var", "metric_cons", "diversity", "pyirt_diffdisc"]:
        data_new[method].append({
            "metric": line["metric"],
            "correlation": line["correlation"],
            "clu": line["clu"][method],
            "cor": line["cor"][method],
        })

# %%
fig_utils.matplotlib_default()
bins = [0, 0.2, 0.4, 0.6, 0.8]
bins_label = bins

def plot_ax(ax, key):
    for method in ["metric_avg", "metric_var", "metric_cons", "diversity", "pyirt_diffdisc"]:
        data_local = sorted(data_new[method], key=lambda x: x["correlation"])
        data_i = np.digitize([abs(v["correlation"]) for v in data_local], bins=bins)
        data_y = collections.defaultdict(list)
        for i, v in zip(data_i, data_local):
            data_y[bins[i]].append(v[key])
        data_y = [
            np.average(data_y[bins[i]])
            for i in range(len(bins))
        ]
        ax.plot(
            bins_label,
            scipy.signal.savgol_filter(data_y, 2, 1),
            label=method,
            linewidth=1.5,
        )
    ax.axhline(
        y=1,
        color="black",
        label="Random",
        linewidth=1.5,
    )
    ax.spines[['top', 'right']].set_visible(False)

fig, axs = plt.subplots(1, 2, figsize=(4, 2.5))
plot_ax(axs[0], "cor")
plot_ax(axs[1], "clu")

# show y-axis as percentage
axs[0].yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1, decimals=0))
axs[1].yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1, decimals=0))

axs[0].set_ylim(0.45, 1.4)
axs[1].set_ylim(0.45, 1.4)

axs[0].set_ylabel("Needed to match correlation", labelpad=0)
axs[1].set_ylabel("Needed to match clusters", labelpad=-3)

# axs[0].set_xticks([0.1, 0.3, 0.5])
# axs[1].set_xticks([0.1, 0.3, 0.5])
axs[1].set_xlabel("Metric correlation with human scores" + " " * 45)
axs[0].tick_params(axis='y', which='major', pad=-1)
# remove tiny tick lines
axs[0].tick_params(left=False)
axs[1].set_yticks(axs[0].get_yticks())
axs[1].set_yticklabels([])
axs[1].tick_params(left=True)



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
    "bleu": "BLEU",
    "chrf": "ChrF",
    "supert": "Supert",
    "unieval_sum": "UniEval",
    "rouge_l_f_score": "ROUGE-L",
    "gpt_sum": "G-Eval",
}

for item in [
    dict(metric="bleu", line_yy=(0.9, 2.0), text_xy=(-0.08, -0.04)),
    dict(metric="supert", line_yy=(0.7, 1.1), text_xy=(-0.15, -0.04)),
    # dict(metric="unieval_sum", line_yy=(0.6, 1.1), text_xy=(-0.09, -0.04)),
    dict(metric="gpt_sum", line_yy=(0.6, 1.1), text_xy=(-0.09, -0.04)),
]:
    metric = item["metric"]
    axs[0].vlines(
        ymin=item["line_yy"][0], ymax=item["line_yy"][1],
        x=metrics_avg[metric],
        color="gray",
        linestyle="--",
        linewidth=0.5,
    )
    axs[0].text(
        x=item["text_xy"][0]+metrics_avg[metric],
        y=item["text_xy"][1]+item["line_yy"][0],
        s=METRIC_NAMES[metric],
        fontsize=8,
        va="center",
        ha="left",
    )
    
for item in [
    dict(metric="chrf", line_yy=(0, 1.3), text_xy=(-0.08, 0.04)),
    dict(metric="supert", line_yy=(0, 1.15), text_xy=(-0.05, 0.04)),
    # dict(metric="unieval_sum", line_yy=(0, 1.05), text_xy=(-0.02, 0.04)),
    dict(metric="gpt_sum", line_yy=(0.5, 0.9), text_xy=(-0.09, 0.04)),
]:
    metric = item["metric"]
    axs[1].vlines(
        ymin=item["line_yy"][0], ymax=item["line_yy"][1],
        x=metrics_avg[metric],
        color="gray",
        linestyle="--",
        linewidth=0.5,
    )
    axs[1].text(
        x=item["text_xy"][0]+metrics_avg[metric],
        y=item["text_xy"][1]+item["line_yy"][1],
        s=METRIC_NAMES[metric],
        fontsize=8,
        va="center",
        ha="left",
    )
    

# separator between subplots
plt.tight_layout(pad=0)
plt.subplots_adjust(wspace=0.2)
plt.savefig("../figures_pdf/17-metric_quality_performance_summeval.pdf")
plt.show()