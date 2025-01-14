# %%
import subset2evaluate.select_subset
import subset2evaluate.evaluate
import subset2evaluate.utils
import random
import tqdm
import numpy as np
import scipy.stats
import subset2evaluate.utils as utils
import utils_fig as fig_utils
import matplotlib.pyplot as plt
import matplotlib as mpl
import collections

random.seed(0)

# use ALL the data
data_old_all = list(utils.load_data_wmt_all(normalize=True).values())

# %%
accs_all = collections.defaultdict(list)
clus_all = collections.defaultdict(list)
corrs_all = []


for data_old in tqdm.tqdm(data_old_all):
    systems = list(data_old[0]["scores"].keys())
    data_y_human = [
        line["scores"][sys]["human"]
        for line in data_old
        for sys in systems
    ]
    metrics = set(list(data_old[0]["scores"].values())[0])
    if "human" not in metrics:
        continue
    metrics.remove("human")
    print(metrics)
    for metric in tqdm.tqdm(metrics):
        try:
            data_y_metric = [
                line["scores"][sys][metric]
                for line in data_old
                for sys in systems
            ]
            corrs_all.append(scipy.stats.pearsonr(data_y_human, data_y_metric)[0])

            data_new_avg = subset2evaluate.select_subset.run_select_subset(data_old, method="random", metric=metric)
            clu_new, acc_new = subset2evaluate.evaluate.eval_cluacc(data_new_avg, data_old)
            clus_all['random'].append(np.average(clu_new))
            accs_all['random'].append(np.average(acc_new))

            data_new_avg = subset2evaluate.select_subset.run_select_subset(data_old, method="metric_avg", metric=metric)
            clu_new, acc_new = subset2evaluate.evaluate.eval_cluacc(data_new_avg, data_old)
            clus_all['avg'].append(np.average(clu_new))
            accs_all['avg'].append(np.average(acc_new))

            data_new_var = subset2evaluate.select_subset.run_select_subset(data_old, method="metric_var", metric=metric)
            clu_new, acc_new = subset2evaluate.evaluate.eval_cluacc(data_new_var, data_old)
            clus_all['var'].append(np.average(clu_new))
            accs_all['var'].append(np.average(acc_new))

            data_new_var = subset2evaluate.select_subset.run_select_subset(data_old, method="diversity_bleu", metric=metric)
            clu_new, acc_new = subset2evaluate.evaluate.eval_cluacc(data_new_var, data_old)
            clus_all['diversity_bleu'].append(np.average(clu_new))
            accs_all['diversity_bleu'].append(np.average(acc_new))

            data_new_irt = subset2evaluate.select_subset.run_select_subset(data_old, method="pyirt_diffdisc", model="4pl_score", epochs=1000, metric=metric, retry_on_error=True)
            clu_new, acc_new = subset2evaluate.evaluate.eval_cluacc(data_new_irt, data_old)
            clus_all['irt'].append(np.average(clu_new))
            accs_all['irt'].append(np.average(acc_new))

        except Exception as e:
            print(e)
            print("Errored on", metric)
            continue


# %%

data_x = np.linspace(0, 0.55, 10)


def aggregate_data_y(data_y):
    assert len(corrs_all) == len(data_y)
    data_y_new = []
    for x1, x2 in zip(data_x, list(data_x[1:]) + [float("inf")]):
        # add all data that are in [x1, x2) interval
        data_y_new.append([y for x, y in zip(corrs_all, data_y) if x1 <= x < x2 and isinstance(y, np.float64)])
    return [np.average(l) for l in data_y_new]


# figure for accuracy
fig_utils.matplotlib_default()

fig, axs = plt.subplots(1, 2, figsize=(4, 2.5))
axs[0].plot(
    data_x,
    aggregate_data_y(accs_all["random"]),
    label="random",
    linewidth=2,
    color="black",
)
axs[0].plot(
    data_x,
    aggregate_data_y(accs_all["diversity_bleu"]),
    label="diversity_bleu",
    linewidth=2,
    color="gray",
)
axs[0].plot(
    data_x,
    aggregate_data_y(accs_all["metric_avg"]),
    label="metric avg",
    linewidth=2,
)
axs[0].plot(
    data_x,
    aggregate_data_y(accs_all["metric_var"]),
    label="metric var",
    linewidth=2,
)

data_y = aggregate_data_y(accs_all["irt"])
axs[0].plot(
    data_x, data_y,
    label="IRT diff.$\\times$disc.",
    linewidth=2,
)

axs[0].set_ylabel("Average accuracy", labelpad=-5)
axs[0].set_xticks([0.0, 0.25, 0.5])
axs[0].set_ylim(None, 1)
axs[0].legend(
    handletextpad=0.4,
    handlelength=0.8,
    labelspacing=0.2,
    facecolor="#ccc",
    loc="upper left",
    fontsize=8
)
# show y-axis as percentage
axs[0].yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1, decimals=0))
axs[0].spines[['top', 'right']].set_visible(False)


# figure for clusters
axs[1].plot(
    data_x,
    aggregate_data_y(clus_all["random"]),
    label="random",
    linewidth=2,
    color="black",
)
axs[1].plot(
    data_x,
    aggregate_data_y(clus_all["diversity_bleu"]),
    label="diversity_bleu",
    linewidth=2,
    color="gray",
)
axs[1].plot(
    data_x,
    aggregate_data_y(clus_all["metric_avg"]),
    label="metric avg",
    linewidth=2,
)
axs[1].plot(
    data_x,
    aggregate_data_y(clus_all["metric_var"]),
    label="metric var",
    linewidth=2,
)

axs[1].plot(
    data_x,
    aggregate_data_y(clus_all["irt"]),
    label="IRT diff.$\\times$disc.",
    linewidth=2,
)

axs[1].set_ylabel("Average cluster count")
axs[1].set_xticks([0.0, 0.25, 0.5])
axs[1].set_xlabel("Metric correlation with human" + " " * 40)
axs[1].spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig("figures_pdf/16-metric_quality_performance.pdf")
plt.show()
