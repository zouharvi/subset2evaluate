# %%

import collections
import subset2evaluate.select_subset
import subset2evaluate.utils as utils
import utils_fig
import numpy as np
import tqdm
import subset2evaluate.evaluate

data_old_all = list(utils.load_data_wmt_test().items())

points_y_cor = collections.defaultdict(list)
points_y_clu = collections.defaultdict(list)

for data_old_name, data_old in tqdm.tqdm(data_old_all):
    for repetitions, method_kwargs in [
        (1, dict(method="metric_var", metric="MetricX-23")),
        (1, dict(method="metric_avg", metric="MetricX-23")),
        (1, dict(method="metric_cons", metric="MetricX-23")),
        (5, dict(method="pyirt_diffdisc", model="4pl_score", metric="MetricX-23", retry_on_error=True)),
        (1, dict(method="diversity", metric="BLEU")),
        (100, dict(method="random")),
    ]:
        points_y_cor_local = []
        points_y_clu_local = []
        for _ in range(repetitions):
            clu_new, cor_new = subset2evaluate.evaluate.eval_clucor(
                subset2evaluate.select_subset.basic(data_old, **method_kwargs),
                data_old,
                metric="human",
            )
            points_y_cor_local.append(cor_new)
            points_y_clu_local.append(clu_new)

        points_y_cor[method_kwargs["method"]].append(points_y_cor_local)
        points_y_clu[method_kwargs["method"]].append(points_y_clu_local)

points_y_cor_raw = points_y_cor.copy()
points_y_clu_raw = points_y_clu.copy()

# %%

points_y_cor_random = np.average(np.array(points_y_cor_raw["random"]), axis=(0))[:10]
points_y_clu_random = np.average(np.array(points_y_clu_raw["random"]), axis=(0))[:10]
points_y_cor_random.sort(axis=0)
points_y_clu_random.sort(axis=0)

import scipy.stats
def confidence_interval(data):
    return scipy.stats.t.interval(
        confidence=0.75,
        df=len(data)-1,
        loc=np.mean(data),
        scale=np.std(data)
    )

# points_y_cor_random = [
#     confidence_interval(np.array(points_y_cor_random)[:, i])
#     for i in range(len(utils.PROPS))
# ]
# points_y_clu_random = [
#     confidence_interval(np.array(points_y_clu_random)[:, i])
#     for i in range(len(utils.PROPS))
# ]

points_y_cor = {
    k: np.average(np.array(v), axis=(0, 1))
    for k, v in points_y_cor_raw.items()
}
points_y_clu = {
    k: np.average(np.array(v), axis=(0, 1))
    for k, v in points_y_clu_raw.items()
}

utils_fig.plot_subset_selection(
    points=[
        (utils.PROPS, points_y_cor["random"], f"Random {np.average(points_y_cor['random']):.1%}"),
        (utils.PROPS, points_y_cor["metric_avg"], f"MetricAvg {np.average(points_y_cor['metric_avg']):.1%}"),
        (utils.PROPS, points_y_cor["metric_var"], f"MetricVar {np.average(points_y_cor['metric_var']):.1%}"),
        (utils.PROPS, points_y_cor["metric_cons"], f"MetricCons {np.average(points_y_cor['metric_cons']):.1%}"),
        (utils.PROPS, points_y_cor["diversity"], f"Diversity {np.average(points_y_cor['diversity']):.1%}"),
        # (utils.PROPS, points_y_cor["pyirt_diffdisc"], f"DiffDisc {np.average(points_y_cor['pyirt_diffdisc']):.1%}"),
    ],
    colors=["#000000"] + utils_fig.COLORS,
    filename="13-main_outputbased",
    height=1.9,
    fn_extra=lambda ax: [ax.plot(
        range(len(utils.PROPS)),
        [np.average(x) for x in l],
        color="#000000",
        alpha=0.15,
        linewidth=1,
        zorder=0,
    ) for l in points_y_cor_random],
    # fn_extra=lambda ax: ax.fill_between(
    #     range(len(utils.PROPS)),
    #     [x[0] for x in points_y_cor_random],
    #     [x[1] for x in points_y_cor_random],
    #     alpha=0.2,
    #     color="#000000",
    #     linewidth=0,
    # ),
)

utils_fig.plot_subset_selection(
    points=[
        (utils.PROPS, points_y_clu["random"], f"Random {np.average(points_y_clu['random']):.2f}"),
        (utils.PROPS, points_y_clu["metric_avg"], f"MetricAvg {np.average(points_y_clu['metric_avg']):.2f}"),
        (utils.PROPS, points_y_clu["metric_var"], f"MetricVar {np.average(points_y_clu['metric_var']):.2f}"),
        (utils.PROPS, points_y_clu["metric_cons"], f"MetricCons {np.average(points_y_clu['metric_cons']):.2f}"),
        (utils.PROPS, points_y_clu['diversity'], f"Diversity {np.average(points_y_clu['diversity']):.2f}"),
        # (utils.PROPS, points_y_clu['pyirt_diffdisc'], f"DiffDisc {np.average(points_y_clu['pyirt_diffdisc']):.2f}"),
    ],
    colors=["#000000"] + utils_fig.COLORS,
    filename="13-main_outputbased",
    height=1.9,
    fn_extra=lambda ax: [ax.plot(
        range(len(utils.PROPS)),
        [np.average(x) for x in l],
        color="#000000",
        alpha=0.5,
        linewidth=0.5,
        zorder=0,
    ) for l in points_y_clu_random],

    # fn_extra=lambda ax: ax.fill_between(
    #     range(len(utils.PROPS)),
    #     [x[0] for x in points_y_clu_random],
    #     [x[1] for x in points_y_clu_random],
    #     alpha=0.2,
    #     color="#000000",
    #     linewidth=0,
    # ),
)
