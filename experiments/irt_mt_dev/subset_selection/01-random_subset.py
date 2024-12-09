# %%

import irt_mt_dev.utils as utils
import irt_mt_dev.utils.fig
import random
import numpy as np
import tqdm
import scipy.stats as st
import os
os.chdir("/home/vilda/irt-mt-dev")
import subset2evaluate.evaluate
import subset2evaluate.select_subset

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

    # repeat each sampling 50 times to smooth it out
    for _ in range(50):
        (_, clu_new), acc_new = subset2evaluate.evaluate.run_evaluate_topk(
            data_old,
            subset2evaluate.select_subset.run_select_subset(data_old, method="random"),
            metric="human"
        )
        points_y_acc.append(acc_new)
        points_y_clu.append(clu_new)
    
    points_y_acc_all.append(np.average(points_y_acc, axis=0))
    points_y_clu_all.append(np.average(points_y_clu, axis=0))

print(f"Average ACC {np.average(points_y_acc):.2%}")
print(f"Average CLU {np.average(points_y_clu):.2f}")


# %%
from importlib import reload
reload(irt_mt_dev.utils.fig)
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
    points=[(utils.PROPS, [np.average(l) for l in np.array(points_y_acc).T], f"Random {np.average(points_y_acc):.2%}")],
    filename="01-random_subset",
    fn_extra=plot_extra_acc,
)
irt_mt_dev.utils.fig.plot_subset_selection(
    points=[(utils.PROPS, [np.average(l) for l in np.array(points_y_clu).T], f"Random {np.average(points_y_clu):.2f}")],
    filename="01-random_subset",
    fn_extra=plot_extra_clu,
)
