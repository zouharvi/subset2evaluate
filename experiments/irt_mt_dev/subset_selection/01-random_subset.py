# %%

import irt_mt_dev.utils as utils
import irt_mt_dev.utils.fig
import random
import numpy as np
import tqdm
import scipy.stats as st
import os
os.chdir("/home/vilda/irt-mt-dev")

data_old_all = list(utils.load_data_wmt_all().values())[:9]

points_y_acc_all = []
points_y_clu_all = []

def confidence_interval(data):
    return st.t.interval(
        confidence=0.9,
        df=len(data)-1,
        loc=np.mean(data),
        scale=np.std(data)
    )

for data_old in tqdm.tqdm(data_old_all):
    points_y_acc = []
    points_y_clu = []
    for prop in utils.PROPS:
        points_y_local_acc = []
        points_y_local_clu = []

        # repeat each sampling 10 times to smooth it out
        for _ in range(10):
            data_new = random.sample(data_old, k=int(len(data_old)*prop))
            points_y_local_acc.append(utils.eval_order_accuracy(data_new, data_old))
            points_y_local_clu.append(utils.eval_system_clusters(data_new))

        points_y_acc.append(np.average(points_y_local_acc))
        points_y_clu.append(np.average(points_y_local_clu))

    points_y_acc_all.append(points_y_acc)
    points_y_clu_all.append(points_y_clu)

print(f"Average ACC {np.average(points_y_acc):.2%}")
print(f"Average CLU {np.average(points_y_clu):.2f}")

# %%
def plot_extra_acc(ax):
    for points_y_acc in points_y_acc_all:
        ax.plot(
            utils.PROPS,
            points_y_acc,
            marker=None,
            color="black",
            linewidth=1,
            alpha=0.2,
        )
def plot_extra_clu(ax):
    for points_y_clu in points_y_clu_all:
        ax.plot(
            utils.PROPS,
            points_y_clu,
            marker=None,
            color="black",
            linewidth=1,
            alpha=0.2,
        )

irt_mt_dev.utils.fig.plot_subset_selection(
    points=[(utils.PROPS, [np.average(l) for l in points_y_acc], f"Random {np.average(points_y_acc):.2%}")],
    filename="01-random_subset",
    fn_extra=plot_extra_acc,
)
irt_mt_dev.utils.fig.plot_subset_selection(
    points=[(utils.PROPS, [np.average(l) for l in points_y_clu], f"Random {np.average(points_y_clu):.2f}")],
    filename="01-random_subset",
    fn_extra=plot_extra_clu,
)
