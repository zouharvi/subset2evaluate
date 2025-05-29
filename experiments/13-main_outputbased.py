# %%

import collections
import subset2evaluate.select_subset
import subset2evaluate.utils as utils
import utils_fig
import numpy as np
import tqdm
import subset2evaluate.evaluate

data_old_all = list(utils.load_data_wmt_test().items())

points_y_spa = collections.defaultdict(list)

for data_old_name, data_old in tqdm.tqdm(data_old_all):
    for repetitions, method_kwargs in [
        (1, dict(method="metric_var", metric="MetricX-23-c")),
        (1, dict(method="metric_avg", metric="MetricX-23-c")),
        (1, dict(method="metric_cons", metric="MetricX-23-c")),
        (5, dict(method="pyirt_diffdisc", model="4pl_score", metric="MetricX-23-c", retry_on_error=True)),
        (1, dict(method="diversity", metric="lm")),
        (100, dict(method="random")),
    ]:
        points_y_spa_local = []
        for _ in range(repetitions):
            spa_new = subset2evaluate.evaluate.eval_spa(
                subset2evaluate.select_subset.basic(data_old, **method_kwargs),
                data_old,
                metric="human",
                props=utils.PROPS,
            )
            points_y_spa_local.append(spa_new)

        points_y_spa[method_kwargs["method"]].append(points_y_spa_local)

points_y_spa_raw = points_y_spa.copy()

# %%
import importlib
importlib.reload(utils_fig)

points_y_spa_random = np.average(np.array(points_y_spa_raw["random"]), axis=(0))
points_y_spa_random.sort(axis=0)
points_y_spa_random_090 = [
    utils.confidence_interval(np.array(points_y_spa_random)[:, i], confidence=0.90)
    for i in range(len(utils.PROPS))
]

points_y_spa = {
    k: np.average(np.array(v), axis=(0, 1))
    for k, v in points_y_spa_raw.items()
}


utils_fig.plot_subset_selection(
    points=[
        (utils.PROPS, points_y_spa["random"], f"Random {np.average(points_y_spa['random']):.1%}"),
        (utils.PROPS, points_y_spa["metric_avg"], f"MetricAvg {np.average(points_y_spa['metric_avg']):.1%}"),
        (utils.PROPS, points_y_spa["metric_var"], f"MetricVar {np.average(points_y_spa['metric_var']):.1%}"),
        (utils.PROPS, points_y_spa["metric_cons"], f"MetricCons {np.average(points_y_spa['metric_cons']):.1%}"),
        (utils.PROPS, points_y_spa['diversity'], f"Diversity {np.average(points_y_spa['diversity']):.1%}"),
        (utils.PROPS, points_y_spa['pyirt_diffdisc'], f"DiffDisc {np.average(points_y_spa['pyirt_diffdisc']):.1%}"),
    ],
    measure="spa",
    colors=["#000000"] + utils_fig.COLORS,
    filename="13-main_outputbased",
    fn_extra=lambda ax: [
        ax.fill_between(
            range(len(utils.PROPS)),
            [x[0] for x in points_y_spa_random_090],
            [x[1] for x in points_y_spa_random_090],
            alpha=0.2,
            color="#000000",
            linewidth=0,
        ),
    ]
)
