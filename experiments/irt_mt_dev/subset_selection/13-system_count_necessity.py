# %%
import subset2evaluate.select_subset
import subset2evaluate.evaluate
import subset2evaluate.utils
import copy
import random
import tqdm
import numpy as np
import multiprocessing
import irt_mt_dev.utils.fig as fig_utils
import matplotlib.pyplot as plt
import os
os.chdir("/home/vilda/irt-mt-dev")

random.seed(0)

data_old = subset2evaluate.utils.load_data("wmt23/en-cs")
data_old_i_to_line = {line["i"]: line for line in data_old}
systems = list(data_old[0]["scores"].keys())

# %%
accs_all = []
clus_all = []
for subset_size in tqdm.tqdm(range(1, len(systems)+1)):
    def _run(_):
        systems_local = random.sample(systems, k=subset_size)
        data_old_local = copy.deepcopy(data_old)
        data_old_local = [
            {
                **line,
                "scores": {
                    sys: v
                    for sys, v in line["scores"].items()
                    if sys in systems_local
                }
            }
            for line in data_old_local
        ]

        data_new_avg = subset2evaluate.select_subset.run_select_subset(data_old_local, method="avg", metric="MetricX-23")
        data_new_var = subset2evaluate.select_subset.run_select_subset(data_old_local, method="var", metric="MetricX-23")
        data_new_irt = subset2evaluate.select_subset.run_select_subset(data_old_local, method="irt_ic", model="scalar", metric="MetricX-23")

        # we dropped some systems but we can recover them with the same ordering from data_old
        (_, clu_new_avg), acc_new_avg = subset2evaluate.evaluate.run_evaluate_topk(data_old, [
            data_old_i_to_line[line["i"]]
            for line in data_new_avg
        ])
        (_, clu_new_var), acc_new_var = subset2evaluate.evaluate.run_evaluate_topk(data_old, [
            data_old_i_to_line[line["i"]]
            for line in data_new_var
        ])
        (_, clu_new_irt), acc_new_irt = subset2evaluate.evaluate.run_evaluate_topk(data_old, [
            data_old_i_to_line[line["i"]]
            for line in data_new_irt
        ])
        return (
            (np.average(clu_new_avg), np.average(clu_new_var), np.average(clu_new_irt)),
            (np.average(acc_new_avg), np.average(acc_new_var), np.average(acc_new_irt))
        )
    
    ITERS = 10
    if subset_size == len(systems):
        ITERS = 1
    results = [_run(_) for _ in range(ITERS)]
    clus_new = [x[0] for x in results]
    accs_new = [x[1] for x in results]

    # NOTE: can't use torch in multiprocessing
    # with multiprocessing.Pool(20) as pool:
    #     acc_new = pool.map(_run, range(ITERS))
    accs_all.append(np.average(accs_new, axis=0))
    clus_all.append(np.average(clus_new, axis=0))

# %%
acc_random = []
clu_random = []
for _ in range(50):
    data_new = subset2evaluate.select_subset.run_select_subset(data_old, method="random")
    (_, clu_new), acc_new = subset2evaluate.evaluate.run_evaluate_topk(data_old, data_new)
    acc_random.append(np.average(acc_new))
    clu_random.append(np.average(clu_new))

# %%
from scipy.signal import savgol_filter

fig_utils.matplotlib_default()
plt.figure(figsize=(3, 2))
plt.plot(
    range(2, len(systems)+1),
    savgol_filter([x[0] for x in accs_all[1:]], 2, 1),
    label="Heuristics avg"
)
plt.plot(
    range(2, len(systems)+1),
    savgol_filter([x[1] for x in accs_all[1:]], 2, 1),
    label="Heuristics var"
)
plt.plot(
    range(2, len(systems)+1),
    savgol_filter([x[2] for x in accs_all[1:]], 2, 1),
    label="IRT FIC"

)
plt.hlines(
    y=np.average(acc_random),
    xmin=2, xmax=len(systems),
    color="black",
    label="Random",
)
# plt.ylim(0.91, None)
plt.ylabel("Average accuracy")
plt.xlabel("Number of systems in training data" + " " * 5)
plt.xticks(range(1, len(systems)+1, 3))
plt.legend(
    handletextpad=0.2,
    handlelength=1,
    labelspacing=0.2,
    facecolor="#ccc",
    loc="upper left",
    fontsize=8
    # scatteryoffsets=[0.5]*len(points),
)
plt.gca().spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig("figures_pdf/13-system_count_necessity_acc.pdf")
plt.show()

# %%
from scipy.signal import savgol_filter

fig_utils.matplotlib_default()
plt.figure(figsize=(3, 2))
plt.plot(
    range(2, len(systems)+1),
    savgol_filter([x[0] for x in clus_all[1:]], 2, 1),
    label="MetricX-23 avg"
)
plt.plot(
    range(2, len(systems)+1),
    savgol_filter([x[1] for x in clus_all[1:]], 2, 1),
    label="MetricX-23 var"
)
plt.plot(
    range(2, len(systems)+1),
    savgol_filter([x[2] for x in clus_all[1:]], 2, 1),
    label="IRT FIC"
)
plt.hlines(
    y=np.average(clu_random),
    xmin=2, xmax=len(systems),
    color="black",
    label="Random",
)
# plt.ylim(0.91, None)
plt.ylabel("Average cluster count" + " "*10, labelpad=10)
plt.xlabel("Number of systems in training data" + " " * 5)
plt.xticks(range(1, len(systems)+1, 3))
# plt.legend()
plt.gca().spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig("figures_pdf/13-system_count_necessity_clu.pdf")
plt.show()