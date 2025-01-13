# %%

import collections
import subset2evaluate.utils as utils
import utils_fig
import subset2evaluate.select_subset
import subset2evaluate.evaluate
import numpy as np
import tqdm

data_old_all = list(utils.load_data_wmt_all().items())[:9]

points_y_acc = collections.defaultdict(list)
points_y_clu = collections.defaultdict(list)


# cache models because that's where we lose a lot of time
MODELS = {
    method: subset2evaluate.select_subset.run_select_subset(data_old_all[0][1], method=method, return_model=True)[1]
    for method in ["precomet_avg", "precomet_var"]
}

for data_old_name, data_old in tqdm.tqdm(data_old_all):
    for method, model in MODELS.items():
        data_new = subset2evaluate.select_subset.run_select_subset(data_old, method=method, load_model=model)
        clu_new, acc_new = subset2evaluate.evaluate.run_evaluate_topk(data_old, data_new, metric="human")
        points_y_acc[method].append(acc_new)
        points_y_clu[method].append(clu_new)

points_y_acc = {
    k: np.average(np.array(v), axis=0)
    for k, v in points_y_acc.items()
}
points_y_clu = {
    k: np.average(np.array(v), axis=0)
    for k, v in points_y_clu.items()
}
# %%
utils_fig.plot_subset_selection(
    points=[
        (utils.PROPS, points_y_acc["precomet_avg"], f"PreCOMET$^\\mathrm{{avg}}$ {np.average(points_y_acc['precomet_avg']):.1%}"),
        (utils.PROPS, points_y_acc["precomet_var"], f"PreCOMET$^\\mathrm{{var}}$ {np.average(points_y_acc['precomet_var']):.1%}"),
    ],
    filename="22-main_sourcebased_precomet",
    colors=utils_fig.COLORS,
)

utils_fig.plot_subset_selection(
    points=[
        (utils.PROPS, points_y_clu["precomet_avg"], f"PreCOMET$^\\mathrm{{avg}}$ {np.average(points_y_clu['precomet_avg']):.2f}"),
        (utils.PROPS, points_y_clu["precomet_var"], f"PreCOMET$^\\mathrm{{var}}$ {np.average(points_y_clu['precomet_var']):.2f}"),
    ],
    filename="22-main_sourcebased_precomet",
    colors=utils_fig.COLORS,
)
