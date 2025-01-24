# %%

import subset2evaluate
import subset2evaluate.utils
import subset2evaluate.evaluate
import numpy as np
import collections
import utils_fig

data_old = subset2evaluate.utils.load_data_summeval(normalize=True)
data_old_metrics = subset2evaluate.utils.load_data("../../subset2evaluate-tmp/data_other/sumeval_gpt.jsonl")
data_old_metrics_i = {
    x["i"]: x
    for x in data_old_metrics
}
assert all(x["i"] in data_old_metrics_i for x in data_old)
for x in data_old:
    x["scores"] = {
        sys: v | data_old_metrics_i[x["i"]]["scores"][sys]
        for sys, v in x["scores"].items()
        if sys in data_old_metrics_i[x["i"]]["scores"]
    }
    x["scores"] = {
        sys: v | {
            "gpt_sum": v["gpt_relevance"] + v["gpt_coherence"] + v["gpt_consistency"] + v["gpt_fluency"],
            "gpt_mul": v["gpt_relevance"] * v["gpt_coherence"] * v["gpt_consistency"] * v["gpt_fluency"],
        }
        for sys, v in x["scores"].items()
    }

PROPS = np.geomspace(0.25, 0.75, 5)

# %%
# parity
for method_kwargs in [
    dict(method="metric_avg", metric="supert"),
    dict(method="metric_var", metric="supert"),
    dict(method="metric_cons", metric="supert"),
    dict(method="diversity", metric="LM"),
    dict(method="pyirt_diffdisc", metric="supert"),
]:
    cor_local = []
    clu_local = []
    for metric_target in ["gpt_relevance", "gpt_coherence", "gpt_consistency", "gpt_fluency", "gpt_sum"]:
        par_clu, par_cor = subset2evaluate.evaluate.eval_clucor_randnorm(
            subset2evaluate.select_subset.basic(data_old, **method_kwargs),
            data_old,
            metric=metric_target,
        )
        cor_local.append(par_cor)
        clu_local.append(par_clu)
    print(method_kwargs["method"], f"COR: {np.average(cor_local):.1%} | CLU: {np.average(clu_local):.1%}")


# %%
cor_all = collections.defaultdict(list)
clu_all = collections.defaultdict(list)
for repetitions, method_kwargs in [
    (100, dict(method="random")),
    (1, dict(method="metric_avg", metric="supert")),
    (1, dict(method="metric_var", metric="supert")),
    (1, dict(method="metric_cons", metric="supert")),
    (1, dict(method="diversity", metric="LM")),
    (5, dict(method="pyirt_diffdisc", metric="supert", retry_on_error=True)),
]:
    for _ in range(repetitions):
        data_new = subset2evaluate.select_subset.basic(data_old, **method_kwargs)
        for metric_target in ["gpt_relevance", "gpt_coherence", "gpt_consistency", "gpt_fluency", "gpt_sum"]:
            clu_new, cor_new = subset2evaluate.evaluate.eval_clucor(data_new, data_old, metric=metric_target, props=PROPS)
            cor_all[method_kwargs['method']].append(cor_new)
            clu_all[method_kwargs['method']].append(clu_new)
    print(method_kwargs["method"], f"COR: {np.average(cor_all[method_kwargs['method']]):.1%} | CLU: {np.average(clu_all[method_kwargs['method']]):.2f}")


# %%
# find best metric: supert
_ = subset2evaluate.evaluate.eval_metrics_correlations(data_old, metric_target="gpt_sum", display=True)


# %%
# table & average

points_y_cor = {
    k: np.average(v, axis=0)
    for k,v in cor_all.items()
}
points_y_clu = {
    k: np.average(v, axis=0)
    for k,v in clu_all.items()
}

for method in points_y_clu.keys():
    print(
        f"{method:>15}",
        f"{np.average(points_y_cor[method]):.1%}",
        f"{np.average(points_y_clu[method]):.2f}",
    )

# %%
# plot

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
    filename="31-summeval_target_gpt",
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
    filename="31-summeval_target_gpt",
    height=1.9,
    ylim=(2, 4.1),
)