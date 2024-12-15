# %%

import collections
import irt_mt_dev.utils as utils
import irt_mt_dev.utils.fig
import subset2evaluate.select_subset
import numpy as np
import tqdm
import os
os.chdir("/home/vilda/irt-mt-dev")

data_old_all = list(utils.load_data_wmt_all().values())[:9]

points_y_acc_all = collections.defaultdict(list)
points_y_clu_all = collections.defaultdict(list)

for data_old in tqdm.tqdm(data_old_all):
    # sort by the heuristic
    data_precomet_avg = subset2evaluate.select_subset.run_select_subset(data_old, method="precomet_avg")
    data_precomet_var = subset2evaluate.select_subset.run_select_subset(data_old, method="precomet_var")

    points_y_acc = collections.defaultdict(list)
    points_y_clu = collections.defaultdict(list)

    for prop in utils.PROPS:

        points_y_acc["precomet_avg"].append(
            utils.eval_subset_accuracy(data_precomet_avg[: int(len(data_old) * prop)], data_old)
        )
        points_y_clu["precomet_avg"].append(
            utils.eval_system_clusters(data_precomet_avg[: int(len(data_old) * prop)])
        )
        points_y_acc["precomet_var"].append(
            utils.eval_subset_accuracy(data_precomet_var[: int(len(data_old) * prop)], data_old)
        )
        points_y_clu["precomet_var"].append(
            utils.eval_system_clusters(data_precomet_var[: int(len(data_old) * prop)])
        )
    
    # add lists to the global list
    for k, v in points_y_acc.items():
        points_y_acc_all[k].append(v)
    for k, v in points_y_clu.items():
        points_y_clu_all[k].append(v)

points_y_acc_all = {
    k: np.average(np.array(v), axis=0)
    for k,v in points_y_acc_all.items()
}
points_y_clu_all = {
    k: np.average(np.array(v), axis=0)
    for k,v in points_y_clu_all.items()
}
# %%
import importlib
importlib.reload(irt_mt_dev.utils.fig)
irt_mt_dev.utils.fig.plot_subset_selection(
    points=[
        (utils.PROPS, points_y_acc_all["precomet_avg"], f"PreCOMET average {np.average(points_y_acc_all['precomet_avg']):.2%}"),
        (utils.PROPS, points_y_acc_all["precomet_var"], f"PreCOMET variance {np.average(points_y_acc_all['precomet_var']):.2%}"),
    ],
    filename="22-precomet_all",
    colors=irt_mt_dev.utils.fig.COLORS[1:],
)


irt_mt_dev.utils.fig.plot_subset_selection(
    points=[
        (utils.PROPS, points_y_clu_all["precomet_avg"], f"PreCOMET average {np.average(points_y_clu_all['precomet_avg']):.2f}"),
        (utils.PROPS, points_y_clu_all["precomet_var"], f"PreCOMET variance {np.average(points_y_clu_all['precomet_var']):.2f}"),
    ],
    filename="22-precomet_all",
    colors=irt_mt_dev.utils.fig.COLORS[1:],
)