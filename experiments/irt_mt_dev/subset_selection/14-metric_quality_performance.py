# %%
import subset2evaluate.select_subset
import subset2evaluate.evaluate
import subset2evaluate.utils
import copy
import random
import tqdm
import numpy as np
import scipy.stats
import irt_mt_dev.utils as utils
import irt_mt_dev.utils.fig as fig_utils
import matplotlib.pyplot as plt
import collections
import os
os.chdir("/home/vilda/irt-mt-dev")

random.seed(0)

data_old_all = list(utils.load_data_wmt_all(normalize=True).values())[:9]
# data_old_i_to_line = {line["i"]: line for line in data_old}
# systems = list(data_old[0]["scores"].keys())

# %%
acc_random = []
clu_random = []

for data_old in data_old_all:
    for _ in range(50):
        data_new = subset2evaluate.select_subset.run_select_subset(data_old, method="random")
        (_, clu_new), acc_new = subset2evaluate.evaluate.run_evaluate_topk(data_old, data_new, metric="human")
        acc_random.append(np.average(acc_new))
        clu_random.append(np.average(clu_new))
print(f"{'random':>25} corr=00.0% | clu={np.average(clu_random):>.2f} | acc={np.average(acc_random):.2f}")

# %%
accs_all = collections.defaultdict(lambda: collections.defaultdict(list))
clus_all = collections.defaultdict(lambda: collections.defaultdict(list))
corrs_all = collections.defaultdict(list)

# get intersection of all metrics
metrics = set(data_old[0]["scores"]["NLLB_MBR_BLEU"].keys())
metrics.remove("human")
for data_old in data_old_all:
    metrics_new = set(data_old[0]["scores"]["NLLB_MBR_BLEU"].keys())
    metrics = metrics.intersection(metrics_new)
metrics = list(metrics)
print(metrics)


for data_old in tqdm.tqdm(data_old_all):
    systems = list(data_old[0]["scores"].keys())
    data_y_human = [
        line["scores"][sys]["human"]
        for line in data_old
        for sys in systems
    ]
    for metric in tqdm.tqdm(metrics):
        data_y_metric = [
            line["scores"][sys][metric]
            for line in data_old
            for sys in systems
        ]
        corrs_all[metric].append(scipy.stats.pearsonr(data_y_human, data_y_metric)[0])
        
        data_new_avg = subset2evaluate.select_subset.run_select_subset(data_old, method="avg", metric=metric)
        (_, clu_new), acc_new = subset2evaluate.evaluate.run_evaluate_topk(data_old, data_new_avg)
        clus_all['avg'][metric].append(np.average(clu_new))
        accs_all['avg'][metric].append(np.average(acc_new))

        data_new_var = subset2evaluate.select_subset.run_select_subset(data_old, method="var", metric=metric)
        (_, clu_new), acc_new = subset2evaluate.evaluate.run_evaluate_topk(data_old, data_new_var)
        clus_all['var'][metric].append(np.average(clu_new))
        accs_all['var'][metric].append(np.average(acc_new))

        data_new_irt = subset2evaluate.select_subset.run_select_subset(data_old, method="pyirt_fic", model="scalar", epochs=1000, metric=metric, retry_on_error=True)
        (_, clu_new), acc_new = subset2evaluate.evaluate.run_evaluate_topk(data_old, data_new_irt)
        clus_all['irt'][metric].append(np.average(clu_new))
        accs_all['irt'][metric].append(np.average(acc_new))

# average across all datasets
corrs_all = {k: np.average(v) for k, v in corrs_all.items()}
clus_all["avg"] = {k: np.average(v) for k, v in clus_all["avg"].items()}
accs_all["avg"] = {k: np.average(v) for k, v in accs_all["avg"].items()}
clus_all["var"] = {k: np.average(v) for k, v in clus_all["var"].items()}
accs_all["var"] = {k: np.average(v) for k, v in accs_all["var"].items()}
clus_all["irt"] = {k: np.average(v) for k, v in clus_all["irt"].items()}
accs_all["irt"] = {k: np.average(v) for k, v in accs_all["irt"].items()}

# %%
fig_utils.matplotlib_default()

# figure for accuracy

plt.figure(figsize=(4, 2))
data_x = [corrs_all[metric] for metric in metrics]
data_y = [accs_all["avg"][metric] for metric in metrics]
plt.scatter(
    data_x,
    data_y,
    label=f"metric avg {scipy.stats.pearsonr(data_x, data_y)[0]:.2f}",
    s=15,
    alpha=0.5,
    linewidth=0,
)
poly1d_fn = np.poly1d(np.polyfit(
    data_x, data_y,
1))
plt.plot(
    sorted(data_x),
    poly1d_fn(sorted(data_x)),
    linestyle='-',
    color=fig_utils.COLORS[0]
)

data_y = [accs_all["var"][metric] for metric in metrics]
plt.scatter(
    data_x,
    data_y,
    label=f"metric var {scipy.stats.pearsonr(data_x, data_y)[0]:.2f}",
    s=15,
    alpha=0.5,
    linewidth=0,
)
poly1d_fn = np.poly1d(np.polyfit(
    data_x, data_y,
1))
plt.plot(
    sorted(data_x),
    poly1d_fn(sorted(data_x)),
    linestyle='-',
    color=fig_utils.COLORS[1]
)

data_y = [accs_all["irt"][metric] for metric in metrics]
plt.scatter(
    data_x, data_y,
    label=f"IRT information {scipy.stats.pearsonr(data_x, data_y)[0]:.2f}",
    s=15,
    alpha=0.5,
    linewidth=0,
)
poly1d_fn = np.poly1d(np.polyfit(
    data_x, data_y,
1))
plt.plot(
    sorted(data_x),
    poly1d_fn(sorted(data_x)),
    linestyle='-',
    color=fig_utils.COLORS[2]
)

plt.hlines(
    y=np.average(acc_random),
    xmin=min(data_x),
    xmax=max(data_x),
    color="black",
    linestyles='dashed',
    label="Random",
)
plt.ylabel("Average accuracy")
plt.xlabel("Metric correlation with human")
plt.legend(
    handletextpad=0.2,
    handlelength=1.5,
    labelspacing=0.2,
    facecolor="#ccc",
    loc="lower right",
    fontsize=8
)
# show y-axis as percentage
import matplotlib as mpl
plt.gca().yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1, decimals=0))
plt.gca().spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig("figures_pdf/14-metric_quality_performance_acc.pdf")
plt.show()

# %%

# figure for clusters

plt.figure(figsize=(4, 2))
data_y = [clus_all["avg"][metric] for metric in metrics]
plt.scatter(
    data_x, data_y,
    label=f"metric avg {scipy.stats.pearsonr(data_x, data_y)[0]:.2f}",
    s=15,
    alpha=0.5,
    linewidth=0,
)
poly1d_fn = np.poly1d(np.polyfit(
    data_x, data_y,
1))
plt.plot(
    sorted(data_x),
    poly1d_fn(sorted(data_x)),
    linestyle='-',
    color=fig_utils.COLORS[0]
)

data_y = [clus_all["var"][metric] for metric in metrics]
plt.scatter(
    data_x, data_y,
    label=f"metric var {scipy.stats.pearsonr(data_x, data_y)[0]:.2f}",
    s=15,
    alpha=0.5,
    linewidth=0,
)
poly1d_fn = np.poly1d(np.polyfit(
    data_x, data_y,
1))
plt.plot(
    sorted(data_x),
    poly1d_fn(sorted(data_x)),
    linestyle='-',
    color=fig_utils.COLORS[1]
)


data_y = [clus_all["irt"][metric] for metric in metrics]
plt.scatter(
    data_x, data_y,
    label=f"IRT information {scipy.stats.pearsonr(data_x, data_y)[0]:.2f}",
    s=15,
    alpha=0.5,
    linewidth=0,
)
poly1d_fn = np.poly1d(np.polyfit(
    data_x, data_y,
1))
plt.plot(
    sorted(data_x),
    poly1d_fn(sorted(data_x)),
    linestyle='-',
    color=fig_utils.COLORS[2]
)


plt.hlines(
    y=np.average(clu_random),
    xmin=min(data_x),
    xmax=max(data_x),
    color="black",
    linestyles='dashed',
    label="Random",
)
plt.ylabel("Average cluster count" + " "*10)
plt.xlabel("Metric correlation with human")
plt.legend(
    handletextpad=0.2,
    handlelength=1.5,
    labelspacing=0.2,
    facecolor="#ccc",
    loc="upper left",
    fontsize=8
)
plt.gca().spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig("figures_pdf/14-metric_quality_performance_clu.pdf")
plt.show()