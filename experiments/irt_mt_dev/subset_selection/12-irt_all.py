# %%

import irt_mt_dev.utils as utils
import irt_mt_dev.utils.fig
import numpy as np
import os
import subset2evaluate.evaluate
import subset2evaluate.select_subset
import tqdm
import collections

os.chdir("/home/vilda/irt-mt-dev")

# data_irt = json.load(open("computed/irt_wmt_4pl_s0_pyirt.json", "r"))[-1]
data_old_all = list(utils.load_data_wmt_all(normalize=True).values())[:9]
points_y_acc_all = collections.defaultdict(list)
points_y_clu_all = collections.defaultdict(list)

for data_old in tqdm.tqdm(data_old_all):

    # the parameters *can't* be re-used probably instead of having to train from scrach
    # because it affects the early stopping criteria
    for method in ["irt_diff", "irt_disc", "irt_feas", "irt_fic"]:
        points_y_acc_local = []
        points_y_clu_local = []

        # run multiple times to smooth
        for _ in range(2):
            (_, clu_new), acc_new = subset2evaluate.evaluate.run_evaluate_topk(
                data_old,
                subset2evaluate.select_subset.run_select_subset(data_old, method=method, metric="MetricX-23", model="scalar"),
                metric="human"
            )
            points_y_acc_local.append(acc_new)
            points_y_clu_local.append(clu_new)

        points_y_acc_all[method].append(np.average(points_y_acc_local, axis=0))
        points_y_clu_all[method].append(np.average(points_y_clu_local, axis=0))

# %%
points_y_acc_all = {
    k: np.average(np.array(v), axis=0)
    for k,v in points_y_acc_all.items()
}
points_y_clu_all = {
    k: np.average(np.array(v), axis=0)
    for k,v in points_y_clu_all.items()
}

# %%
irt_mt_dev.utils.fig.plot_subset_selection(
    [
        (utils.PROPS, points_y_acc_all['irt_feas'], f"IRT feasability {np.average(points_y_acc_all['irt_feas']):.2%}"),
        (utils.PROPS, points_y_acc_all['irt_diff'], f"IRT difficulty {np.average(points_y_acc_all['irt_diff']):.2%}"),
        (utils.PROPS, points_y_acc_all['irt_disc'], f"IRT discriminability {np.average(points_y_acc_all['irt_disc']):.2%}"),
        (utils.PROPS, points_y_acc_all['irt_fic'], f"IRT information content {np.average(points_y_acc_all['irt_fic']):.2%}"),
    ],
    "12-irt_all",
)
irt_mt_dev.utils.fig.plot_subset_selection(
    [
        (utils.PROPS, points_y_clu_all['irt_feas'], f"IRT feasability {np.average(points_y_clu_all['irt_feas']):.2f}"),
        (utils.PROPS, points_y_clu_all['irt_diff'], f"IRT difficulty {np.average(points_y_clu_all['irt_diff']):.2f}"),
        (utils.PROPS, points_y_clu_all['irt_disc'], f"IRT discriminability {np.average(points_y_clu_all['irt_disc']):.2f}"),
        (utils.PROPS, points_y_clu_all['irt_fic'], f"IRT information content {np.average(points_y_clu_all['irt_fic']):.2f}"),
    ],
    "12-irt_all",
)