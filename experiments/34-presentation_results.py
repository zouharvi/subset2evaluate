# %%


import subset2evaluate.utils as utils
import utils_fig
import numpy as np
import subset2evaluate.evaluate
import subset2evaluate.select_subset

data_old_all = list(utils.load_data_wmt_all().values())[:9]
data_old = data_old_all[1]

cor_random = []
for seed in range(100):
    _, cor_new = subset2evaluate.evaluate.eval_clucor(
        data_old,
        subset2evaluate.select_subset.basic(data_old, method="random", seed=seed),
        metric="human"
    )
    cor_random.append(cor_new)

# %%

_, cor_metric_var = subset2evaluate.evaluate.eval_clucor(
    subset2evaluate.select_subset.basic(data_old, method="metric_var", metric="MetricX-23"),
    data_old,
    metric="human"
)
_, cor_metric_avg = subset2evaluate.evaluate.eval_clucor(
    subset2evaluate.select_subset.basic(data_old, method="metric_avg", metric="MetricX-23"),
    data_old,
    metric="human"
)
_, cor_diffdisc = subset2evaluate.evaluate.eval_clucor(
    subset2evaluate.select_subset.basic(data_old, method="pyirt_diffdisc", metric="MetricX-23", retry_on_error=True),
    data_old,
    metric="human",
)
_, cor_precomet_diffdisc = subset2evaluate.evaluate.eval_clucor(
    subset2evaluate.select_subset.basic(data_old, method="precomet_diffdisc"),
    data_old,
    metric="human",
)
_, cor_precomet_var = subset2evaluate.evaluate.eval_clucor(
    subset2evaluate.select_subset.basic(data_old, method="precomet_var"),
    data_old,
    metric="human",
)

# %%
utils_fig.plot_subset_selection(
    points=[
        (utils.PROPS, np.average(cor_random, axis=0), f"Random {np.average(cor_random):.1%}"),
    ],
    colors=[
        "black",
    ],
    filename="34-presentation_results_0"
)
utils_fig.plot_subset_selection(
    points=[
        (utils.PROPS, np.average(cor_random, axis=0), f"Random {np.average(cor_random):.1%}"),
        (utils.PROPS, cor_metric_avg, f"Metric avg {np.average(cor_metric_avg):.1%}"),
    ],
    colors=[
        "black",
        utils_fig.COLORS[0],
    ],
    filename="34-presentation_results_1"
)

utils_fig.plot_subset_selection(
    points=[
        (utils.PROPS, np.average(cor_random, axis=0), f"Random {np.average(cor_random):.1%}"),
        (utils.PROPS, cor_metric_var, f"Metric var {np.average(cor_metric_var):.1%}"),
    ],
    colors=[
        "black",
        utils_fig.COLORS[0],
    ],
    filename="34-presentation_results_2"
)

utils_fig.plot_subset_selection(
    points=[
        (utils.PROPS, np.average(cor_random, axis=0), f"Random {np.average(cor_random):.1%}"),
        (utils.PROPS, cor_metric_var, f"Metric var {np.average(cor_metric_var[:-1]):.1%}"),
        (utils.PROPS, cor_diffdisc, f"IRT {np.average(cor_diffdisc):.1%}"),
    ],
    colors=[
        "black",
        utils_fig.COLORS[0],
        utils_fig.COLORS[1],
    ],
    filename="34-presentation_results_3"
)

utils_fig.plot_subset_selection(
    points=[
        (utils.PROPS, np.average(cor_random, axis=0), f"Random {np.average(cor_random):.1%}"),
        (utils.PROPS, cor_precomet_diffdisc, f"PreCOMET$^\\mathrm{{IRT}}$ {np.average(cor_precomet_diffdisc):.1%}"),
        (utils.PROPS, cor_precomet_var, f"PreCOMET$^\\mathrm{{var}}$ {np.average(cor_precomet_var[:-1]):.1%}"),
    ],
    colors=[
        "black",
        utils_fig.COLORS[0],
        utils_fig.COLORS[2],
        utils_fig.COLORS[3],
    ],
    filename="34-presentation_results_4"
)
