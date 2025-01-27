# %%
import subset2evaluate.select_subset
import subset2evaluate.evaluate
import subset2evaluate.utils
import tqdm
import numpy as np
import scipy.stats
import subset2evaluate.utils as utils
import utils_fig as fig_utils
import matplotlib.pyplot as plt
import matplotlib as mpl
import collections
import pickle

# use all WMT data
data_old_all = list(utils.load_data_wmt_all(normalize=True).values())

# %%
cors_all = collections.defaultdict(list)
clus_all = collections.defaultdict(list)
corrs_all = []
corrs_all_named = collections.defaultdict(list)

for data_old in tqdm.tqdm(data_old_all):
    models = list(data_old[0]["scores"].keys())
    data_y_human = [
        line["scores"][model]["human"]
        for line in data_old
        for model in models
    ]
    metrics = set(list(data_old[0]["scores"].values())[0])
    if "human" not in metrics:
        continue
    metrics.remove("human")
    print(metrics)
    for metric in tqdm.tqdm(list(metrics)):
        try:
            data_y_metric = [
                line["scores"][model][metric]
                for line in data_old
                for model in models
            ]
            corrs_all.append(scipy.stats.pearsonr(data_y_human, data_y_metric)[0])
            corrs_all_named[metric].append(scipy.stats.pearsonr(data_y_human, data_y_metric)[0])

            data_new_avg = subset2evaluate.select_subset.basic(data_old, method="random", metric=metric)
            clu_new, cor_new = subset2evaluate.evaluate.eval_clucor(data_new_avg, data_old)
            clus_all['random'].append(np.average(clu_new))
            cors_all['random'].append(np.average(cor_new))

            data_new_avg = subset2evaluate.select_subset.basic(data_old, method="metric_avg", metric=metric)
            clu_new, cor_new = subset2evaluate.evaluate.eval_clucor(data_new_avg, data_old)
            clus_all['metric_avg'].append(np.average(clu_new))
            cors_all['metric_avg'].append(np.average(cor_new))

            data_new_var = subset2evaluate.select_subset.basic(data_old, method="metric_var", metric=metric)
            clu_new, cor_new = subset2evaluate.evaluate.eval_clucor(data_new_var, data_old)
            clus_all['metric_var'].append(np.average(clu_new))
            cors_all['metric_var'].append(np.average(cor_new))

            data_new_var = subset2evaluate.select_subset.basic(data_old, method="diversity", metric="BLEU")
            clu_new, cor_new = subset2evaluate.evaluate.eval_clucor(data_new_var, data_old)
            clus_all['diversity'].append(np.average(clu_new))
            cors_all['diversity'].append(np.average(cor_new))

            data_new_irt = subset2evaluate.select_subset.basic(data_old, method="pyirt_diffdisc", model="4pl_score", metric=metric, retry_on_error=True)
            clu_new, cor_new = subset2evaluate.evaluate.eval_clucor(data_new_irt, data_old)
            clus_all['pyirt_diffdisc'].append(np.average(clu_new))
            cors_all['pyirt_diffdisc'].append(np.average(cor_new))

            data_new_ali = subset2evaluate.select_subset.basic(data_old, method="metric_cons", metric=metric)
            clu_new, cor_new = subset2evaluate.evaluate.eval_clucor(data_new_ali, data_old)
            clus_all['metric_cons'].append(np.average(clu_new))
            cors_all['metric_cons'].append(np.average(cor_new))

        except Exception as e:
            print(e)
            print("Errored on", metric)
            continue


corrs_all_named = {
    k: np.average(v)
    for k, v in corrs_all_named.items()
}

# %%
# backup
with open("../computed/16-metric_quality_performance.pkl", "wb") as f:
    pickle.dump((cors_all, clus_all, corrs_all), f)


# %%
data_old_all = list(utils.load_data_wmt_all(normalize=True).items())
data_old_count = 0
prev_name = None
for (data_old_name, langs), data_old in data_old_all:
    if prev_name != data_old_name:
        print(prev_name, data_old_count)
        prev_name = data_old_name

    metrics = list(list(data_old[0]["scores"].values())[0].keys())
    metrics = [m for m in metrics if m != "human"]
    data_old_count += len(metrics)

# %%
# temp hack
# load
with open("../computed/16-metric_quality_performance.pkl", "rb") as f:
    cors_all, clus_all, corrs_all = pickle.load(f)

clus_all["metric_cons"] = clus_all["metric_alignment"]
cors_all["metric_cons"] = cors_all["metric_alignment"]
clus_all["diversity"] = clus_all["diversity_bleu"]
cors_all["diversity"] = cors_all["diversity_bleu"]

corrs_all_named = collections.defaultdict(list)
for data_old in data_old_all:
    # align to WMT23, WMT24, WMT22
    if sum(len(y) for y in corrs_all_named.values()) >= 561:
        break

    models = list(data_old[0]["scores"].keys())
    data_y_human = [
        line["scores"][model]["human"]
        for line in data_old
        for model in models
    ]
    metrics = set(list(data_old[0]["scores"].values())[0])
    if "human" not in metrics:
        continue
    metrics.remove("human")
    for metric in list(metrics):
        data_y_metric = [
            line["scores"][model][metric]
            for line in data_old
            for model in models
        ]
        corrs_all_named[metric].append(scipy.stats.pearsonr(data_y_human, data_y_metric)[0])

corrs_all_named = {
    k: np.average(v)
    for k, v in corrs_all_named.items()
}


# %%

METRIC_NAMES = {
    "BLEU": "BLEU",
    "chrF": "ChrF",
    "BLEURT-20": "BLEURT",
    "BERTscore": "BERTscore",
    "prismRef": "Prism",
    "MetricX-23": "MetricX",
}

data_x = [0, 0.12, 0.24, 0.39, 0.49, 1]
# constrain to WMT23, WMT24, WMT22
corrs_all = corrs_all[:561]

def aggregate_data_y(data_y):
    # assert len(corrs_all) == len(data_y)
    data_y_new = []
    for x1, x2 in zip(data_x, data_x[1:]):
        # add all data that are in [x1, x2) interval
        data_y_new.append([y for x, y in zip(corrs_all, data_y) if x1 <= x < x2 and isinstance(y, np.float64)])
    return [np.average(l) for l in data_y_new]

aggregate_data_y_cors = {
    k: aggregate_data_y(v)
    for k, v in cors_all.items()
}
aggregate_data_y_clus = {
    k: aggregate_data_y(v)
    for k, v in clus_all.items()
}

fig_utils.matplotlib_default()

# defaut line width
mpl.rcParams["lines.linewidth"] = 1.5

fig, axs = plt.subplots(1, 2, figsize=(4, 2.5))

# figure for clusters
axs[1].plot(
    data_x[:-1],
    aggregate_data_y_clus["random"],
    label="Random",
    color="black",
    zorder=100,
)
axs[1].plot(
    data_x[:-1],
    aggregate_data_y_clus["metric_avg"],
    label="MetricAvg",
)
axs[1].plot(
    data_x[:-1],
    aggregate_data_y_clus["metric_var"],
    label="MetricVar",
)
axs[1].plot(
    data_x[:-1],
    aggregate_data_y_clus["metric_cons"],
    label="MetricCons",
)
axs[1].plot(
    data_x[:-1],
    aggregate_data_y_clus["diversity"],
    label="Diversity",
)
axs[1].plot(
    data_x[:-1],
    aggregate_data_y_clus["pyirt_diffdisc"],
    label="DiffDisc",
)

axs[1].set_ylabel("Cluster count", labelpad=-1)
axs[1].set_xticks(
    data_x[:-1],
    [f"{x+0.1:.2f}"[:3] for x in data_x[:-1]],
)
# set max number of yticks
axs[1].yaxis.set_major_locator(plt.MaxNLocator(6))
axs[1].set_xlabel("Metric correlation with human" + " " * 40)
axs[1].spines[['top', 'right']].set_visible(False)


# plot vertical lines for some special metrics
for text_xy, text_top, relpos, line_yy, metric in [
    ((0.005, 3.55), True, (1, 0.5), (2.85, 3.55), "BLEU"),
    ((0.13, 3.65), True, (0.5, 0.5), (2.85, 3.65), "chrF"),
    ((0.22, 2.70), False, (0, 0.5), (3.00, 3.75), "BERTscore"),
    ((0.303, 2.80), False, (0.5, 0.5), (3.00, 3.75), "MetricX-23"),
]:
    
    axs[1].vlines(
        ymin=line_yy[0], ymax=line_yy[1],
        x=corrs_all_named[metric],
        color="gray",
        linestyle="--",
        linewidth=0.5,
    )
    axs[1].annotate(
        METRIC_NAMES[metric],
        xy=(corrs_all_named[metric], line_yy[1] if text_top else line_yy[0]),
        xytext=(text_xy[0], text_xy[1]),
        arrowprops=dict(
            arrowstyle="-",
            linestyle="--",
            color="gray",
            lw=0.5,
            shrinkA=0.0, shrinkB=0.0,
            relpos=relpos,
        ),
        fontsize=8,
        va="center",
    )


axs[0].plot(
    data_x[:-1],
    aggregate_data_y_cors["random"],
    label="Random",
    color="black",
    zorder=100
)
axs[0].plot(
    data_x[:-1],
    aggregate_data_y_cors["metric_avg"],
    label="MetricAvg",
)
axs[0].plot(
    data_x[:-1],
    aggregate_data_y_cors["metric_var"],
    label="MetricVar",
)
axs[0].plot(
    data_x[:-1],
    aggregate_data_y_cors["metric_cons"],
    label="MetricCons",
)
axs[0].plot(
    data_x[:-1],
    aggregate_data_y_cors["diversity"],
    label="Diversity",
)
axs[0].plot(
    data_x[:-1],
    aggregate_data_y_cors["pyirt_diffdisc"],
    label="DiffDisc",
)

axs[0].set_ylabel("Rank correlation", labelpad=-1)
axs[0].set_xticks(
    data_x[:-1],
    [f"{x+0.1:.2f}"[:3] for x in data_x[:-1]],
)

# show y-axis as percentage
axs[0].yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1, decimals=0))
axs[0].spines[['top', 'right']].set_visible(False)

axs[0].set_ylim(0.88, None)



# plot vertical lines for some special metrics
for text_xy, text_top, relpos, line_yy, metric in [
    ((0.01, 0.935), True, (1.0, 0.5), (0.89, 0.935), "BLEU"),
    ((0.13, 0.945), True, (0.5, 0.5), (0.89, 0.945), "chrF"),
    ((0.22, 0.894), False, (0.1, 1), (0.915, 0.95), "BERTscore"),
    ((0.305, 0.90), False, (0.5, 1), (0.902, 0.95), "MetricX-23"),
]:
    
    axs[0].vlines(
        ymin=line_yy[0], ymax=line_yy[1],
        x=corrs_all_named[metric],
        color="gray",
        linestyle="--",
        linewidth=0.5,
    )
    axs[0].annotate(
        METRIC_NAMES[metric],
        xy=(corrs_all_named[metric], line_yy[1] if text_top else line_yy[0]),
        xytext=(text_xy[0], text_xy[1]),
        arrowprops=dict(
            arrowstyle="-",
            linestyle="--",
            color="gray",
            lw=0.5,
            shrinkA=0.0, shrinkB=0.0,
            relpos=relpos,
        ),
        fontsize=8,
        va="center",
    )

# legend is done manually in LaTeX
# axs[1].legend(
#     handletextpad=0.4,
#     handlelength=0.8,
#     labelspacing=0.2,
#     facecolor="#ddd",
#     loc="upper right",
#     bbox_to_anchor=(0.8, 1.2),
#     ncol=3,
#     fontsize=9,
# )
# plt.subplots_adjust(right=5.5)


plt.tight_layout(pad=0.1)
plt.savefig("../figures_pdf/16-metric_quality_performance.pdf")
plt.show()