# %%

import subset2evaluate
import subset2evaluate.evaluate
import numpy as np
import collections
import utils_fig

data_old = subset2evaluate.utils.load_data_summeval(normalize=True)
PROPS = np.geomspace(0.25, 0.75, 10)

# %%
# parity
for method_kwargs in [
    dict(method="metric_var", metric="supert"),
    dict(method="metric_avg", metric="supert"),
    dict(method="pyirt_diffdisc", metric="supert"),
    dict(method="diversity", metric="BLEU"),
]:
    par_clu, par_cor = subset2evaluate.evaluate.eval_clucor_randnorm(
        subset2evaluate.select_subset.basic(data_old, **method_kwargs),
        data_old,
        metric="human_sum",
    )
    print(method_kwargs["method"], f"COR: {par_cor:.1%} | CLU: {par_clu:.1%}")

# %%

cor_all = collections.defaultdict(list)
clu_all = collections.defaultdict(list)
for metric_target in ["human_relevance", "human_coherence", "human_consistency", "human_fluency", "human_sum"]:
    for repetitions, method_kwargs in [
        (100, dict(method="random")),
        (1, dict(method="metric_cons", metric="supert")),
        (1, dict(method="metric_avg", metric="supert")),
        (1, dict(method="metric_var", metric="supert")),
        (1, dict(method="diversity", metric="LM")),
        (5, dict(method="pyirt_diffdisc", metric="supert", retry_on_error=True)),
    ]:
        for _ in range(repetitions):
            data_new = subset2evaluate.select_subset.basic(data_old, **method_kwargs)
            clu_new, cor_new = subset2evaluate.evaluate.eval_clucor(data_new, data_old, metric=metric_target, props=PROPS)
            cor_all[method_kwargs['method']].append(cor_new)
            clu_all[method_kwargs['method']].append(clu_new)
        print(method_kwargs["method"], f"COR: {np.average(cor_all[method_kwargs['method']]):.1%} | CLU: {np.average(clu_all[method_kwargs['method']]):.2f}")


# %%
# kmeans special
cor_all = []
clu_all = []
load_model = None
for prop in subset2evaluate.utils.PROPS:
    B = int(len(data_old) * prop)
    data_new, load_model = subset2evaluate.select_subset.basic(
        data_old,
        method="kmeans", budget=B,
        features="src",
        load_model=load_model, return_model=True,
    )
    data_new = data_new[:B]
    cor_new = subset2evaluate.evaluate.eval_subset_correlation(data_new, data_old, metric="human_sum")
    clu_new = subset2evaluate.evaluate.eval_subset_clusters(data_new, metric="human_sum")

    clu_all.append(clu_new)
    cor_all.append(cor_new)
print("kmeans", f"COR: {np.average(cor_all):.1%} | CLU: {np.average(clu_all):.2f}")

# %%
# find best metric: supert

# chrf
# density
# coverage
# coverage
_ = subset2evaluate.evaluate.eval_metrics_correlations(data_old, metric_target="human_fluency", display=True)

# %%
# plot

points_y_cor = {
    k: np.average(v, axis=0)
    for k,v in cor_all.items()
}
points_y_clu = {
    k: np.average(v, axis=0)
    for k,v in clu_all.items()
}

utils_fig.plot_subset_selection(
    points=[
        (PROPS, points_y_cor["random"], f"Random {np.average(points_y_cor['random']):.1%}"),
        (PROPS, points_y_cor["metric_avg"], f"MetricAvg {np.average(points_y_cor['metric_avg']):.1%}"),
        (PROPS, points_y_cor["metric_var"], f"MetricVar {np.average(points_y_cor['metric_var']):.1%}"),
        (PROPS, points_y_cor["metric_cons"], f"MetricCons {np.average(points_y_cor['metric_cons']):.1%}"),
        (PROPS, points_y_cor["diversity"], f"Diversity {np.average(points_y_cor['diversity']):.1%}"),
        (PROPS, points_y_cor["pyirt_diffdisc"], f"DiffDisc {np.average(points_y_cor['pyirt_diffdisc']):.1%}"),
    ],
    colors=["#000000"] + utils_fig.COLORS,
    filename="30-summeval",
    height=1.9,
    ylim=(0.9, 1),
)

utils_fig.plot_subset_selection(
    points=[
        (PROPS, points_y_clu["random"], f"Random {np.average(points_y_clu['random']):.2f}"),
        (PROPS, points_y_clu["metric_avg"], f"MetricAvg {np.average(points_y_clu['metric_avg']):.2f}"),
        (PROPS, points_y_clu["metric_var"], f"MetricVar {np.average(points_y_clu['metric_var']):.2f}"),
        (PROPS, points_y_clu["metric_cons"], f"MetricCons {np.average(points_y_clu['metric_cons']):.2f}"),
        (PROPS, points_y_clu['diversity'], f"Diversity {np.average(points_y_clu['diversity']):.2f}"),
        (PROPS, points_y_clu['pyirt_diffdisc'], f"DiffDisc {np.average(points_y_clu['pyirt_diffdisc']):.2f}"),
    ],
    colors=["#000000"] + utils_fig.COLORS,
    filename="30-summeval",
    height=1.9,
    ylim=(2, 4.1),
)