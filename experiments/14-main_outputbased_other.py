# %%

import subset2evaluate.utils as utils
import utils_fig
import numpy as np
import subset2evaluate.evaluate
import subset2evaluate.select_subset
import tqdm
import collections
import importlib
importlib.reload(subset2evaluate.select_subset)

data_old_all = list(utils.load_data_wmt_all(normalize=True).items())[:9]
points_y_acc_all = collections.defaultdict(lambda: collections.defaultdict(list))
points_y_clu_all = collections.defaultdict(lambda: collections.defaultdict(list))

for data_name, data_old in tqdm.tqdm(data_old_all):
    # run multiple times to smooth
    for _ in range(5):
        data_new = subset2evaluate.select_subset.run_select_subset(
            data_old, method="pyirt_diffdisc",
            metric="MetricX-23-c", model="4pl_score", epochs=1000,
            retry_on_error=True,
        )
        clu_new, acc_new = subset2evaluate.evaluate.run_evaluate_topk(
            data_old, data_new,
            metric="human"
        )
    
        points_y_acc_all["pyirt_diffdisc"][data_name].append(acc_new)
        points_y_clu_all["pyirt_diffdisc"][data_name].append(clu_new)


    for _ in range(1):
        data_new = subset2evaluate.select_subset.run_select_subset(
            data_old, method="diversity",
        )
        clu_new, acc_new = subset2evaluate.evaluate.run_evaluate_topk(
            data_old, data_new,
            metric="human"
        )
    
        points_y_acc_all["diversity"][data_name].append(acc_new)
        points_y_clu_all["diversity"][data_name].append(clu_new)
# %%

points_y_acc_all_backup = points_y_acc_all
points_y_clu_all_backup = points_y_clu_all

# %%
points_y_acc_all = {
    method: np.average(np.array(list(method_v.values())), axis=(0, 1))
    for method, method_v in points_y_acc_all.items()
}
points_y_clu_all = {
    method: np.average(np.array(list(method_v.values())), axis=(0, 1))
    for method, method_v in points_y_clu_all.items()
}


# %%

utils_fig.plot_subset_selection(
    [
        # (utils.PROPS, points_y_acc_all['pyirt_feas'], f"IRT feasability {np.average(points_y_acc_all['pyirt_feas']):.2%}"),
        # (utils.PROPS, points_y_acc_all['pyirt_diff'], f"difficulty {np.average(points_y_acc_all['pyirt_diff']):.2%}"),
        # (utils.PROPS, points_y_acc_all['pyirt_disc'], f"discriminability {np.average(points_y_acc_all['pyirt_disc']):.2%}"),
        (utils.PROPS, points_y_acc_all['pyirt_diffdisc'], f"IRT diff.$\\times$disc. {np.average(points_y_acc_all['pyirt_diffdisc']):.2%}"),
        # (utils.PROPS, points_y_acc_all['pyirt_fic'], f"information {np.average(points_y_acc_all['pyirt_fic']):.2%}"),
        # (utils.PROPS, points_y_acc_all['pyirt_experiment'], f"experiment {np.average(points_y_acc_all['pyirt_experiment']):.2%}"),
        (utils.PROPS, points_y_acc_all['diversity'], f"Diversity {np.average(points_y_acc_all['diversity']):.2%}"),
    ],
    "14-main_outputbased_other",
)
utils_fig.plot_subset_selection(
    [
        # (utils.PROPS, points_y_clu_all['pyirt_feas'], f"IRT feasability {np.average(points_y_clu_all['pyirt_feas']):.2f}"),
        # (utils.PROPS, points_y_clu_all['pyirt_diff'], f"difficulty {np.average(points_y_clu_all['pyirt_diff']):.2f}"),
        # (utils.PROPS, points_y_clu_all['pyirt_disc'], f"discriminability {np.average(points_y_clu_all['pyirt_disc']):.2f}"),
        (utils.PROPS, points_y_clu_all['pyirt_diffdisc'], f"IRT diff.$\\times$disc. {np.average(points_y_clu_all['pyirt_diffdisc']):.2f}"),
        # (utils.PROPS, points_y_clu_all['pyirt_fic'], f"information {np.average(points_y_clu_all['pyirt_fic']):.2f}"),
        # (utils.PROPS, points_y_clu_all['pyirt_experiment'], f"experiment {np.average(points_y_clu_all['pyirt_experiment']):.2f}"),
        (utils.PROPS, points_y_clu_all['diversity'], f"Diversity {np.average(points_y_clu_all['diversity']):.2f}"),
    ],
    "14-main_outputbased_other",
)