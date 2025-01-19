# %%
import subset2evaluate.select_subset
import subset2evaluate.evaluate
import subset2evaluate.utils
import copy
import random
import tqdm
import numpy as np
import subset2evaluate.utils as utils
import utils_fig as fig_utils
import matplotlib.pyplot as plt

random.seed(0)

data_old_all = list(utils.load_data_wmt_all(normalize=True).values())[:9]

# %%
accs_all = []
clus_all = []
points_x = list(range(1, 16))
for subset_size in tqdm.tqdm(points_x):
    clus_new = []
    accs_new = []
    for data_old in data_old_all:
        data_old_i_to_line = {line["i"]: line for line in data_old}
        models = list(data_old[0]["scores"].keys())

        # run multiple times
        for _ in range(4):
            models_local = random.sample(models, k=min(subset_size, len(models)))
            data_old_local = copy.deepcopy(data_old)
            data_old_local = [
                {
                    **line,
                    "scores": {
                        model: v
                        for model, v in line["scores"].items()
                        if model in models_local
                    }
                }
                for line in data_old_local
            ]

            data_new_avg = subset2evaluate.select_subset.basic(data_old_local, method="metric_avg", metric="MetricX-23-c")
            data_new_var = subset2evaluate.select_subset.basic(data_old_local, method="metric_var", metric="MetricX-23-c")
            data_new_div = subset2evaluate.select_subset.basic(data_old_local, method="diversity_bleu")
            data_new_irt = subset2evaluate.select_subset.basic(data_old_local, method="pyirt_diffdisc", model="4pl_score", metric="MetricX-23-c", retry_on_error=True)

            # we dropped some models but we can recover them with the same ordering from data_old
            clu_new_avg, cor_new_avg = subset2evaluate.evaluate.eval_clucor(
                [
                    data_old_i_to_line[line["i"]]
                    for line in data_new_avg
                ],
                data_old,
            )
            clu_new_var, cor_new_var = subset2evaluate.evaluate.eval_clucor(
                [
                    data_old_i_to_line[line["i"]]
                    for line in data_new_var
                ],
                data_old,
            )
            clu_new_div, cor_new_div = subset2evaluate.evaluate.eval_clucor(
                [
                    data_old_i_to_line[line["i"]]
                    for line in data_new_div
                ],
                data_old,
            )
            clu_new_irt, cor_new_irt = subset2evaluate.evaluate.eval_clucor(
                [
                    data_old_i_to_line[line["i"]]
                    for line in data_new_irt
                ],
                data_old,
            )
            clus_new.append((clu_new_avg, clu_new_var, clu_new_irt, clu_new_div))
            accs_new.append((cor_new_avg, cor_new_var, cor_new_irt, cor_new_div))

    accs_all.append(np.average(accs_new, axis=0))
    clus_all.append(np.average(clus_new, axis=0))

# %%
acc_random = []
clu_random = []
for data_old in data_old_all:
    for _ in range(10):
        data_new = subset2evaluate.select_subset.basic(data_old, method="random")
        clu_new, cor_new = subset2evaluate.evaluate.eval_clucor(data_new, data_old)
        acc_random.append(np.average(cor_new))
        clu_random.append(np.average(clu_new))

# %%

plot_kwargs = dict(
    marker="o",
    markersize=5,
    linewidth=2,
)

fig_utils.matplotlib_default()
plt.figure(figsize=(4, 2.5))
plt.plot(
    points_x,
    [np.average(x[0]) for x in accs_all],
    label="MetricX avg.",
    **plot_kwargs
)
plt.plot(
    points_x[1:],
    [np.average(x[1]) for x in accs_all[1:]],
    label="MetricX var.",
    **plot_kwargs
)
plt.plot(
    points_x,
    [np.average(x[2]) for x in accs_all],
    label="IRT diff$\\times$disc",
    **plot_kwargs
)
# TODO: this is incorrect, diversity shouldn't be constant!
# should be fixed by also filtering the "tgt"
plt.hlines(
    y=np.average(accs_all[0][3]),
    xmin=min(points_x,), xmax=max(points_x,),
    label="Diversity",
    color=fig_utils.COLORS[3],
)
plt.hlines(
    y=np.average(acc_random),
    xmin=min(points_x,), xmax=max(points_x,),
    color="black",
    label="Random",
)
# plt.ylim(0.77, 1)
plt.ylabel("Average accuracy")
plt.xlabel("Number of models in training data" + " " * 5)
plt.xticks(range(min(points_x), max(points_x), 3))
plt.legend(
    handletextpad=0.2,
    handlelength=1,
    labelspacing=0.2,
    facecolor="#ccc",
    loc="lower right",
    fontsize=8,
)
plt.gca().spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig("figures_pdf/15-model_count_necessity_acc.pdf")
plt.show()

# %%

fig_utils.matplotlib_default()
plt.figure(figsize=(4, 2.5))
plt.plot(
    points_x,
    [np.average(x[0]) for x in clus_all],
    label="MetricX-23 avg.",
    **plot_kwargs
)
plt.plot(
    points_x[1:],
    [np.average(x[1]) for x in clus_all[1:]],
    label="MetricX-23 var.",
    **plot_kwargs
)
plt.plot(
    points_x,
    [np.average(x[2]) for x in clus_all],
    label="IRT diff$\\times$disc",
    **plot_kwargs
)
# TODO: this is incorrect, diversity shouldn't be constant!
# should be fixed by also filtering the "tgt"
plt.hlines(
    y=np.average(clus_all[0][3]),
    xmin=min(points_x,), xmax=max(points_x,),
    label="Diversity",
    color=fig_utils.COLORS[3],
)
plt.hlines(
    y=np.average(clu_random),
    xmin=min(points_x,), xmax=max(points_x,),
    color="black",
    label="Random",
)
# plt.ylim(0.91, None)
plt.ylabel("Average cluster count" + " " * 10, labelpad=10)
plt.xlabel("Number of models in training data" + " " * 5)
plt.xticks(range(min(points_x), max(points_x), 3))
plt.legend(
    handletextpad=0.2,
    handlelength=1,
    labelspacing=0.2,
    facecolor="#ccc",
    loc="lower right",
    fontsize=8,
)
plt.gca().spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig("figures_pdf/15-model_count_necessity_clu.pdf")
plt.show()
