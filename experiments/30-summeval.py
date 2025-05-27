# %%

import subset2evaluate
import subset2evaluate.utils
import subset2evaluate.evaluate
import subset2evaluate.select_subset
import numpy as np
import collections
import utils_fig

data_old = subset2evaluate.utils.load_data_summeval(normalize=True, load_extra=True)
PROPS = np.linspace(0.25, 0.75, 10)

# %%
# parity
for method_kwargs in [
    dict(method="metric_avg"),
    dict(method="metric_var"),
    dict(method="metric_cons"),
    dict(method="diversity", metric="LM"),
    dict(method="pyirt_diffdisc", retry_on_error=True),
]:
    spa_local = []
    for metric_target in ["human_relevance", "human_coherence", "human_consistency", "human_fluency", "human_sum"]:
        metric_train = "gpt_" + metric_target.split("_")[1]
        par_spa = subset2evaluate.evaluate.eval_spa_par_randnorm(
            subset2evaluate.select_subset.basic(data_old, **({"metric": metric_train} | method_kwargs)),
            data_old,
            metric=metric_target,
            props=PROPS,
        )
        spa_local.append(par_spa)
    print(method_kwargs["method"], f"SPA: {np.average(spa_local):.1%}")

# %%
spa_all = collections.defaultdict(list)

for metric_target in ["human_relevance", "human_coherence", "human_consistency", "human_fluency", "human_sum"]:
    metric_train = "gpt_" + metric_target.split("_")[1]
    print(metric_target)
    for repetitions, method_kwargs in [
        (100, dict(method="random")),
        (1, dict(method="metric_cons", metric=metric_train)),
        (1, dict(method="metric_avg", metric=metric_train)),
        (1, dict(method="metric_var", metric=metric_train)),
        (1, dict(method="diversity", metric="LM")),
        (5, dict(method="pyirt_disc", metric=metric_train, retry_on_error=True)),
    ]:
        for _ in range(repetitions):
            data_new = subset2evaluate.select_subset.basic(data_old, **method_kwargs)
            spa_new = subset2evaluate.evaluate.eval_spa(data_new, data_old, metric=metric_target, props=PROPS)
            spa_all[method_kwargs['method']].append(spa_new)
        print(method_kwargs["method"], f"SPA: {np.average(spa_all[method_kwargs['method']]):.1%}")
    print()

# %%
# kmeans special
spa_all = []
clu_all = []
load_model = None
for prop in PROPS:
    B = int(len(data_old) * prop)
    data_new, load_model = subset2evaluate.select_subset.basic(
        data_old,
        method="kmeans", budget=B,
        features="src",
        load_model=load_model, return_model=True,
    )
    data_new = data_new[:B]
    spa_new = subset2evaluate.evaluate.eval_subset_correlation(data_new, data_old, metric="human_sum")
    clu_new = subset2evaluate.evaluate.eval_subset_clusters(data_new, metric="human_sum")

    clu_all.append(clu_new)
    spa_all.append(spa_new)
print("kmeans", f"COR: {np.average(spa_all):.1%} | CLU: {np.average(clu_all):.2f}")

# %%
# find best metric: supert
_ = subset2evaluate.evaluate.eval_metrics_correlations(data_old, metric_target="human_sum", display=True)

# %%
# plot
import subset2evaluate.utils as utils

points_y_spa = spa_all

points_y_spa_random = points_y_spa["random"]
points_y_spa_random = np.array(points_y_spa_random)
# post-hoc fixing shape of the array but works out well
points_y_spa_random = points_y_spa_random.reshape(-1, 100, len(PROPS))
points_y_spa_random = np.average(points_y_spa_random, axis=(0))
points_y_spa_random.sort(axis=0)

points_y_spa_random_075 = [
    utils.confidence_interval(points_y_spa_random[:, i], confidence=0.75)
    for i in range(len(PROPS))
]
points_y_spa_random_095 = [
    utils.confidence_interval(points_y_spa_random[:, i], confidence=0.95)
    for i in range(len(PROPS))
]

points_y_spa = {
    k: np.average(np.array(v), axis=(0))
    for k, v in points_y_spa.items()
}


utils_fig.plot_subset_selection(
    points=[
        (PROPS, points_y_spa["random"], f"Random {np.average(points_y_spa['random']):.1%}"),
        (PROPS, points_y_spa["metric_avg"], f"MetricAvg {np.average(points_y_spa['metric_avg']):.1%}"),
        (PROPS, points_y_spa["metric_var"], f"MetricVar {np.average(points_y_spa['metric_var']):.1%}"),
        (PROPS, points_y_spa["metric_cons"], f"MetricCons {np.average(points_y_spa['metric_cons']):.1%}"),
        (PROPS, points_y_spa["diversity"], f"Diversity {np.average(points_y_spa['diversity']):.1%}"),
        (PROPS, points_y_spa["pyirt_disc"], f"DiffDisc {np.average(points_y_spa['pyirt_disc']):.1%}"),
    ],
    measure="spa",
    colors=["#000000"] + utils_fig.COLORS,
    filename="30-summeval",
    ylim=(0.89, 0.98),
    fn_extra=lambda ax: [
        ax.fill_between(
            range(len(PROPS)),
            [x[0] for x in points_y_spa_random_095],
            [x[1] for x in points_y_spa_random_095],
            alpha=0.2,
            color="#000000",
            linewidth=0,
        ),

        ax.fill_between(
            range(len(PROPS)),
            [x[0] for x in points_y_spa_random_075],
            [x[1] for x in points_y_spa_random_075],
            alpha=0.4,
            color="#000000",
            linewidth=0,
        ),
    ]
)