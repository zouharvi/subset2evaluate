# %%
import subset2evaluate.select_subset
import subset2evaluate.evaluate
import subset2evaluate.utils
import copy
import random
import tqdm
import numpy as np
import scipy.stats
import irt_mt_dev.utils.fig as fig_utils
import matplotlib.pyplot as plt
import collections
import os
os.chdir("/home/vilda/irt-mt-dev")

random.seed(0)

data_old = subset2evaluate.utils.load_data("wmt23/en-cs")
data_old_i_to_line = {line["i"]: line for line in data_old}
systems = list(data_old[0]["scores"].keys())

# %%
acc_random = []
clu_random = []

for _ in range(50):
    data_new = subset2evaluate.select_subset.run_select_subset(data_old, method="random")
    (_, clu_new), acc_new = subset2evaluate.evaluate.run_evaluate_topk(data_old, data_new)
    acc_random.append(np.average(acc_new))
    clu_random.append(np.average(clu_new))

# %%
accs_all = collections.defaultdict(list)
clus_all = collections.defaultdict(list)
corrs_all = []

metrics = list(data_old[0]["scores"]["NLLB_MBR_BLEU"].keys())
metrics.remove("human")

print(f"{'random':>25} corr=00.0% | clu={np.average(clu_random):>.2f} | acc={np.average(acc_random):.2f}")

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
    corrs_all.append(scipy.stats.pearsonr(data_y_human, data_y_metric)[0])
    
    data_new_avg = subset2evaluate.select_subset.run_select_subset(data_old, method="avg", metric=metric)
    (_, clu_new), acc_new = subset2evaluate.evaluate.run_evaluate_topk(data_old, [
        data_old_i_to_line[line["i"]]
        for line in data_new_avg
    ])
    clus_all['avg'].append(np.average(clu_new))
    accs_all['avg'].append(np.average(acc_new))

    data_new_var = subset2evaluate.select_subset.run_select_subset(data_old, method="var", metric=metric)
    (_, clu_new), acc_new = subset2evaluate.evaluate.run_evaluate_topk(data_old, [
        data_old_i_to_line[line["i"]]
        for line in data_new_var
    ])
    clus_all['var'].append(np.average(clu_new))
    accs_all['var'].append(np.average(acc_new))


    data_new_var = subset2evaluate.select_subset.run_select_subset(data_old, method="irt_fic", model="scalar", metric=metric)
    (_, clu_new), acc_new = subset2evaluate.evaluate.run_evaluate_topk(data_old, [
        data_old_i_to_line[line["i"]]
        for line in data_new_var
    ])
    clus_all['irt'].append(np.average(clu_new))
    accs_all['irt'].append(np.average(acc_new))

# ['human', 'mre-score-labse-regular', 'MetricX-23', 'chrF', 'COMET', 'f200spBLEU', 'tokengram_F', 'YiSi-1', 'embed_llama', 'XCOMET-XXL', 'BLEU', 'prismRef', 'eBLEU', 'XCOMET-XL', 'MetricX-23-c', 'XCOMET-Ensemble', 'BERTscore', 'XLsim', 'BLEURT-20', 'MetricX-23-b']


# %%
fig_utils.matplotlib_default()
plt.figure(figsize=(3, 2))
plt.scatter(
    corrs_all,
    accs_all["avg"],
    label="metric avg",
    s=10,
)
poly1d_fn = np.poly1d(np.polyfit(corrs_all, accs_all["avg"], 1))
plt.plot(
    sorted(corrs_all),
    poly1d_fn(sorted(corrs_all)),
    linestyle='-',
    color=fig_utils.COLORS[0]
)

plt.scatter(
    corrs_all,
    accs_all["var"],
    label="metric var",
    s=10,
)
poly1d_fn = np.poly1d(np.polyfit(corrs_all, accs_all["var"], 1))
plt.plot(
    sorted(corrs_all),
    poly1d_fn(sorted(corrs_all)),
    linestyle='-',
    color=fig_utils.COLORS[1]
)

plt.scatter(
    corrs_all,
    accs_all["irt"],
    label="IRT FIC",
    s=10,
)
poly1d_fn = np.poly1d(np.polyfit(corrs_all, accs_all["irt"], 1))
plt.plot(
    sorted(corrs_all),
    poly1d_fn(sorted(corrs_all)),
    linestyle='-',
    color=fig_utils.COLORS[2]
)

plt.ylim(0.9, None)
# plt.hlines(
#     y=np.average(acc_random),
#     xmin=min(corrs_all), xmax=max(corrs_all),
#     color="black",
#     label="Random",
# )
plt.ylabel("Average accuracy")
plt.xlabel("Metric correlation with human")
plt.legend(
    handletextpad=0.2,
    handlelength=1,
    labelspacing=0.2,
    facecolor="#ccc",
    loc="upper left",
    fontsize=8
)
plt.gca().spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig("figures_pdf/14-metric_quality_performance_acc.pdf")
plt.show()

# %%
fig_utils.matplotlib_default()
plt.figure(figsize=(3, 2))
plt.scatter(
    corrs_all,
    clus_all["avg"],
    label="metric avg",
    s=10,
)
poly1d_fn = np.poly1d(np.polyfit(corrs_all, clus_all["avg"], 1))
plt.plot(
    sorted(corrs_all),
    poly1d_fn(sorted(corrs_all)),
    linestyle='-',
    color=fig_utils.COLORS[0]
)

plt.scatter(
    corrs_all,
    clus_all["var"],
    label="metric var",
    s=10,
)
poly1d_fn = np.poly1d(np.polyfit(corrs_all, clus_all["var"], 1))
plt.plot(
    sorted(corrs_all),
    poly1d_fn(sorted(corrs_all)),
    linestyle='-',
    color=fig_utils.COLORS[1]
)


plt.scatter(
    corrs_all,
    clus_all["irt"],
    label="IRT FIC",
    s=10,
)
poly1d_fn = np.poly1d(np.polyfit(corrs_all, clus_all["irt"], 1))
plt.plot(
    sorted(corrs_all),
    poly1d_fn(sorted(corrs_all)),
    linestyle='-',
    color=fig_utils.COLORS[2]
)


# plt.hlines(
#     y=np.average(acc_random),
#     xmin=min(corrs_all), xmax=max(corrs_all),
#     color="black",
#     label="Random",
# )
plt.ylabel("Average cluster count" + " "*10)
plt.xlabel("Metric correlation with human")
# plt.legend(
#     handletextpad=0.2,
#     handlelength=1,
#     labelspacing=0.2,
#     facecolor="#ccc",
#     loc="upper left"
# )
plt.gca().spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig("figures_pdf/14-metric_quality_performance_clu.pdf")
plt.show()