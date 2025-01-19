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
    method: subset2evaluate.select_subset.basic(data_old_all[0][1], method=method, return_model=True)[1]
    for method in ["precomet_avg", "precomet_var"]
}
MODELS["random"] = None

for data_old_name, data_old in tqdm.tqdm(data_old_all):
    for repetitions, method in [(1, "precomet_avg"), (1, "precomet_var"), (100, "random")]:
        for _ in range(repetitions):
            data_new = subset2evaluate.select_subset.basic(data_old, method=method, load_model=MODELS[method])
            clu_new, cor_new = subset2evaluate.evaluate.eval_clucor(data_new, data_old, metric="human")
            points_y_acc[method].append(cor_new)
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
        (utils.PROPS, points_y_acc["random"], f"Random {np.average(points_y_acc['random']):.1%}"),
        (utils.PROPS, points_y_acc["precomet_avg"], f"MetricAvg$^\\mathrm{{src}}$ {np.average(points_y_acc['precomet_avg']):.1%}"),
        (utils.PROPS, points_y_acc["precomet_var"], f"MetricVar$^\\mathrm{{src}}$ {np.average(points_y_acc['precomet_var']):.1%}"),
    ],
    colors=["black"] + utils_fig.COLORS,
    filename="22-main_sourcebased_metrics_moment",
)

utils_fig.plot_subset_selection(
    points=[
        (utils.PROPS, points_y_clu["random"], f"Random {np.average(points_y_clu['random']):.2f}"),
        (utils.PROPS, points_y_clu["precomet_avg"], f"MetricAvg$^\\mathrm{{src}}$ {np.average(points_y_clu['precomet_avg']):.2f}"),
        (utils.PROPS, points_y_clu["precomet_var"], f"MetricVar$^\\mathrm{{src}}$ {np.average(points_y_clu['precomet_var']):.2f}"),
    ],
    colors=["black"] + utils_fig.COLORS,
    filename="22-main_sourcebased_metrics_moment",
)
