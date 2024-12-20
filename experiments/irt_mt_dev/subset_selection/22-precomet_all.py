# %%

import collections
import irt_mt_dev.utils as utils
import irt_mt_dev.utils.fig
import subset2evaluate.select_subset
import subset2evaluate.evaluate
import numpy as np
import tqdm
import os
os.chdir("/home/vilda/irt-mt-dev")

data_old_all = list(utils.load_data_wmt_all().items())[:9]

points_y_acc = collections.defaultdict(list)
points_y_clu = collections.defaultdict(list)

for data_old_name, data_old in tqdm.tqdm(data_old_all):
    # train on human data from other languages, just not the one we are evaluating
    data_train = [
        line
        for name, data_old in data_old_all
        if name != data_old_name
        for line in data_old
    ]
    data_premlp_avg = subset2evaluate.select_subset.run_select_subset(data_old, data_train=data_train, method="premlp_avg")
    data_premlp_var = subset2evaluate.select_subset.run_select_subset(data_old, data_train=data_train, method="premlp_var")
    data_precomet_avg = subset2evaluate.select_subset.run_select_subset(data_old, method="precomet_avg")
    data_precomet_var = subset2evaluate.select_subset.run_select_subset(data_old, method="precomet_var")

    clu_new, acc_new = subset2evaluate.evaluate.run_evaluate_topk(data_old, data_precomet_avg)
    points_y_acc["precomet_avg"].append(acc_new)
    points_y_clu["precomet_avg"].append(clu_new)

    clu_new, acc_new = subset2evaluate.evaluate.run_evaluate_topk(data_old, data_precomet_var)
    points_y_acc["precomet_var"].append(acc_new)
    points_y_clu["precomet_var"].append(clu_new)

    clu_new, acc_new = subset2evaluate.evaluate.run_evaluate_topk(data_old, data_premlp_avg)
    points_y_acc["premlp_avg"].append(acc_new)
    points_y_clu["premlp_avg"].append(clu_new)

    clu_new, acc_new = subset2evaluate.evaluate.run_evaluate_topk(data_old, data_premlp_var)
    points_y_acc["premlp_var"].append(acc_new)
    points_y_clu["premlp_var"].append(clu_new)


points_y_acc = {
    k: np.average(np.array(v), axis=0)
    for k,v in points_y_acc.items()
}
points_y_clu = {
    k: np.average(np.array(v), axis=0)
    for k,v in points_y_clu.items()
}
# %%
irt_mt_dev.utils.fig.plot_subset_selection(
    points=[
        (utils.PROPS, points_y_acc["precomet_avg"], f"PreCOMET avg. {np.average(points_y_acc['precomet_avg']):.2%}"),
        (utils.PROPS, points_y_acc["precomet_var"], f"PreCOMET var. {np.average(points_y_acc['precomet_var']):.2%}"),
        (utils.PROPS, points_y_acc["premlp_avg"], f"PreMLP avg. {np.average(points_y_acc['premlp_avg']):.2%}"),
        (utils.PROPS, points_y_acc["premlp_var"], f"PreMLP var. {np.average(points_y_acc['premlp_var']):.2%}"),
    ],
    filename="22-precomet_all",
    colors=irt_mt_dev.utils.fig.COLORS,
)


irt_mt_dev.utils.fig.plot_subset_selection(
    points=[
        (utils.PROPS, points_y_clu["precomet_avg"], f"PreCOMET avg. {np.average(points_y_clu['precomet_avg']):.2f}"),
        (utils.PROPS, points_y_clu["precomet_var"], f"PreCOMET var. {np.average(points_y_clu['precomet_var']):.2f}"),
        (utils.PROPS, points_y_clu["premlp_avg"], f"PreMLP avg. {np.average(points_y_clu['premlp_avg']):.2f}"),
        (utils.PROPS, points_y_clu["premlp_var"], f"PreMLP var. {np.average(points_y_clu['premlp_var']):.2f}"),
    ],
    filename="22-precomet_all",
    colors=irt_mt_dev.utils.fig.COLORS,
)