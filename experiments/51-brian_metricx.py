# %%

import subset2evaluate.utils
import subset2evaluate.evaluate
import collections
import numpy as np

data_all = list(subset2evaluate.utils.load_data_wmt_all().values())[:9]

# %%
points_y_cor = collections.defaultdict(list)
points_y_clu = collections.defaultdict(list)
points_y_cor_auto = collections.defaultdict(list)
points_y_clu_auto = collections.defaultdict(list)

for data_old in data_all:
    for repetitions, method_kwargs in [
        (1, dict(method="metric_var", metric="MetricX-23")),
        (1, dict(method="metric_avg", metric="MetricX-23")),
        (1, dict(method="metric_cons", metric="MetricX-23")),
        (1, dict(method="diversity", metric="BLEU")),
        (50, dict(method="random")),
    ]:
        for _ in range(repetitions):
            data_new = subset2evaluate.select_subset.basic(data_old, **method_kwargs)
            clu_new, cor_new = subset2evaluate.evaluate.eval_clu_cor(
                data_new, data_old,
                metric="human",
            )
            points_y_cor[method_kwargs["method"]].append(cor_new)
            points_y_clu[method_kwargs["method"]].append(clu_new)

            clu_new, cor_new = subset2evaluate.evaluate.eval_clu_cor(
                data_new, data_old,
                metric=("MetricX-23", "human"),
            )
            points_y_cor_auto[method_kwargs["method"]].append(cor_new)
            points_y_clu_auto[method_kwargs["method"]].append(clu_new)

# %%

points_y_cor = {
    k: np.average(np.array(v), axis=0)
    for k, v in points_y_cor.items()
}
points_y_clu = {
    k: np.average(np.array(v), axis=0)
    for k, v in points_y_clu.items()
}
points_y_cor_auto = {
    k: np.average(np.array(v), axis=0)
    for k, v in points_y_cor_auto.items()
}
points_y_clu_auto = {
    k: np.average(np.array(v), axis=0)
    for k, v in points_y_clu_auto.items()
}

# %%

import utils_fig
import subset2evaluate.utils as utils

utils_fig.plot_subset_selection(
    points=[
        (utils.PROPS, points_y_cor["random"], f"Random {np.average(points_y_cor['random']):.1%}"),
        (utils.PROPS, points_y_cor["metric_avg"], f"MetricAvg {np.average(points_y_cor['metric_avg']):.1%}"),
        (utils.PROPS, points_y_cor["metric_var"], f"MetricVar {np.average(points_y_cor['metric_var']):.1%}"),
        (utils.PROPS, points_y_cor["metric_cons"], f"MetricCons {np.average(points_y_cor['metric_cons']):.1%}"),
        (utils.PROPS, points_y_cor["diversity"], f"Diversity {np.average(points_y_cor['diversity']):.1%}"),
    ],
    colors=["#000000"] + utils_fig.COLORS,
    filename="51-brian_metrix_human",
    height=1.9,
    ylim=(0.8, 1.0),
)

utils_fig.plot_subset_selection(
    points=[
        (utils.PROPS, points_y_clu["random"], f"Random {np.average(points_y_clu['random']):.2f}"),
        (utils.PROPS, points_y_clu["metric_avg"], f"MetricAvg {np.average(points_y_clu['metric_avg']):.2f}"),
        (utils.PROPS, points_y_clu["metric_var"], f"MetricVar {np.average(points_y_clu['metric_var']):.2f}"),
        (utils.PROPS, points_y_clu["metric_cons"], f"MetricCons {np.average(points_y_clu['metric_cons']):.2f}"),
        (utils.PROPS, points_y_clu['diversity'], f"Diversity {np.average(points_y_clu['diversity']):.2f}"),
    ],
    colors=["#000000"] + utils_fig.COLORS,
    filename="51-brian_metrix_human",
    height=1.9,
    ylim=(1, 8),
)


utils_fig.plot_subset_selection(
    points=[
        (utils.PROPS, points_y_cor_auto["random"], f"Random {np.average(points_y_cor_auto['random']):.1%}"),
        (utils.PROPS, points_y_cor_auto["metric_avg"], f"MetricAvg {np.average(points_y_cor_auto['metric_avg']):.1%}"),
        (utils.PROPS, points_y_cor_auto["metric_var"], f"MetricVar {np.average(points_y_cor_auto['metric_var']):.1%}"),
        (utils.PROPS, points_y_cor_auto["metric_cons"], f"MetricCons {np.average(points_y_cor_auto['metric_cons']):.1%}"),
        (utils.PROPS, points_y_cor_auto["diversity"], f"Diversity {np.average(points_y_cor_auto['diversity']):.1%}"),
    ],
    colors=["#000000"] + utils_fig.COLORS,
    filename="51-brian_metrix_auto",
    height=1.9,
    ylim=(0.8, 1.0),
)

utils_fig.plot_subset_selection(
    points=[
        (utils.PROPS, points_y_clu_auto["random"], f"Random {np.average(points_y_clu_auto['random']):.2f}"),
        (utils.PROPS, points_y_clu_auto["metric_avg"], f"MetricAvg {np.average(points_y_clu_auto['metric_avg']):.2f}"),
        (utils.PROPS, points_y_clu_auto["metric_var"], f"MetricVar {np.average(points_y_clu_auto['metric_var']):.2f}"),
        (utils.PROPS, points_y_clu_auto["metric_cons"], f"MetricCons {np.average(points_y_clu_auto['metric_cons']):.2f}"),
        (utils.PROPS, points_y_clu_auto['diversity'], f"Diversity {np.average(points_y_clu_auto['diversity']):.2f}"),
    ],
    colors=["#000000"] + utils_fig.COLORS,
    filename="51-brian_metrix_auto",
    height=1.9,
    ylim=(1, 8),
)