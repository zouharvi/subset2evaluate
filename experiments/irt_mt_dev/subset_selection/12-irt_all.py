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
data_old_all = list(utils.load_data_wmt_all().values())[:9]
points_y_acc_all = collections.defaultdict(list)
points_y_clu_all = collections.defaultdict(list)

for data_old in tqdm.tqdm(data_old_all):
    # TODO: run multiple times to smooth?
    (_, clu_new), acc_new = subset2evaluate.evaluate.run_evaluate_topk(
        data_old,
        subset2evaluate.select_subset.run_select_subset(data_old, method="irt_diff", metric="MetricX-23", model="scalar"),
        metric="human"
    )
    points_y_acc_all["diff"].append(acc_new)
    points_y_clu_all["diff"].append(clu_new)

    (_, clu_new), acc_new = subset2evaluate.evaluate.run_evaluate_topk(
        data_old,
        subset2evaluate.select_subset.run_select_subset(data_old, method="irt_disc", metric="MetricX-23", model="scalar"),
        metric="human"
    )
    points_y_acc_all["disc"].append(acc_new)
    points_y_clu_all["disc"].append(clu_new)

    (_, clu_new), acc_new = subset2evaluate.evaluate.run_evaluate_topk(
        data_old,
        subset2evaluate.select_subset.run_select_subset(data_old, method="irt_feas", metric="MetricX-23", model="scalar"),
        metric="human"
    )
    points_y_acc_all["feas"].append(acc_new)
    points_y_clu_all["feas"].append(clu_new)

    (_, clu_new), acc_new = subset2evaluate.evaluate.run_evaluate_topk(
        data_old,
        subset2evaluate.select_subset.run_select_subset(data_old, method="irt_fic", metric="MetricX-23", model="scalar"),
        metric="human"
    )
    points_y_acc_all["fic"].append(acc_new)
    points_y_clu_all["fic"].append(clu_new)

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
        (utils.PROPS, points_y_acc_all['feas'], f"IRT feasability {np.average(points_y_acc_all['feas']):.2%}"),
        (utils.PROPS, points_y_acc_all['diff'], f"IRT difficulty {np.average(points_y_acc_all['diff']):.2%}"),
        (utils.PROPS, points_y_acc_all['disc'], f"IRT discriminability {np.average(points_y_acc_all['disc']):.2%}"),
        (utils.PROPS, points_y_acc_all['fic'], f"IRT information content {np.average(points_y_acc_all['fic']):.2%}"),
    ],
    "12-irt_all",
)
irt_mt_dev.utils.fig.plot_subset_selection(
    [
        (utils.PROPS, points_y_clu_all['feas'], f"IRT feasability {np.average(points_y_clu_all['feas']):.2f}"),
        (utils.PROPS, points_y_clu_all['diff'], f"IRT difficulty {np.average(points_y_clu_all['diff']):.2f}"),
        (utils.PROPS, points_y_clu_all['disc'], f"IRT discriminability {np.average(points_y_clu_all['disc']):.2f}"),
        (utils.PROPS, points_y_clu_all['fic'], f"IRT information content {np.average(points_y_clu_all['fic']):.2f}"),
    ],
    "12-irt_all",
)