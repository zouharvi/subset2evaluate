# %%

import collections
import subset2evaluate.select_subset
import subset2evaluate.utils as utils
import utils_fig
import numpy as np
import tqdm
import subset2evaluate.evaluate

data_old_all = list(utils.load_data_wmt_all().values())[:9]

points_y_acc = collections.defaultdict(list)
points_y_clu = collections.defaultdict(list)


for data_old in tqdm.tqdm(data_old_all):
    for repetitions, method_kwargs in [
        (1, dict(method="metric_var", metric="MetricX-23")),
        (1, dict(method="metric_avg", metric="MetricX-23")),
        (100, dict(method="random")),
    ]:
        for _ in range(repetitions):
            clu_new, acc_new = subset2evaluate.evaluate.eval_cluacc(
                subset2evaluate.select_subset.basic(data_old, **method_kwargs),
                data_old,
                metric="human",
            )
            points_y_acc[method_kwargs["method"]].append(acc_new)
            points_y_clu[method_kwargs["method"]].append(clu_new)

# %%

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
        (utils.PROPS, points_y_acc["metric_avg"], f"MetricX average {np.average(points_y_acc['metric_avg']):.1%}"),
        (utils.PROPS, points_y_acc["metric_var"], f"MetricX variance {np.average(points_y_acc['metric_var']):.1%}"),
    ],
    colors=["black"] + utils_fig.COLORS,
    filename="13-main_outputbased_metrics_moment",
)

utils_fig.plot_subset_selection(
    points=[
        (utils.PROPS, points_y_clu["random"], f"Random {np.average(points_y_clu['random']):.2f}"),
        (utils.PROPS, points_y_clu["metric_avg"], f"MetricX avg. {np.average(points_y_clu['metric_avg']):.2f}"),
        (utils.PROPS, points_y_clu["metric_var"], f"MetricX var. {np.average(points_y_clu['metric_var']):.2f}"),
    ],
    colors=["black"] + utils_fig.COLORS,
    filename="13-main_outputbased_metrics_moment",
)
