# %%

import subset2evaluate.utils as utils
import utils_fig
import numpy as np
import tqdm
import scipy.stats as st
import subset2evaluate.evaluate
import subset2evaluate.select_subset

data_old_all = list(utils.load_data_wmt_all().values())[:9]

points_y_acc_all = []
points_y_clu_all = []


def confidence_interval(data):
    return st.t.interval(
        confidence=0.9,
        df=len(data) - 1,
        loc=np.mean(data),
        scale=np.std(data)
    )


for data_old in tqdm.tqdm(data_old_all):
    points_y_acc = []
    points_y_clu = []

    # repeat each sampling 100 times to smooth it out
    for _ in range(100):
        clu_new, acc_new = subset2evaluate.evaluate.run_evaluate_topk(
            data_old,
            subset2evaluate.select_subset.run_select_subset(data_old, method="random"),
            metric="human"
        )
        points_y_acc.append(acc_new)
        points_y_clu.append(clu_new)

    points_y_acc_all.append(np.average(points_y_acc, axis=0))
    points_y_clu_all.append(np.average(points_y_clu, axis=0))

print(f"Average ACC {np.average(points_y_acc):.1%}")
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


utils_fig.plot_subset_selection(
    points=[(utils.PROPS, [np.average(l) for l in np.array(points_y_acc).T], f"Random {np.average(points_y_acc):.1%}")],
    filename="07-random_subset",
    fn_extra=plot_extra_acc,
)
utils_fig.plot_subset_selection(
    points=[(utils.PROPS, [np.average(l) for l in np.array(points_y_clu).T], f"Random {np.average(points_y_clu):.2f}")],
    filename="07-random_subset",
    fn_extra=plot_extra_clu,
)
