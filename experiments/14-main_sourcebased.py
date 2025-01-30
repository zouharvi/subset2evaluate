# %%
import collections
import tqdm
import subset2evaluate.utils as utils
import subset2evaluate.evaluate
import subset2evaluate.select_subset
import numpy as np
import utils_fig

data_old_all = list(utils.load_data_wmt_all(normalize=True).items())[:9]

# %%

points_y_cor = collections.defaultdict(list)
points_y_clu = collections.defaultdict(list)

# cache models because that's where we lose a lot of time
MODELS = {
    method: subset2evaluate.select_subset.basic(data_old_all[0][1], method=method, return_model=True)[1]
    for method in [
        "cometsrc_avg",
        "cometsrc_var",
        "local_cometsrc_diffdisc",
        "cometsrc_diversity",
        "local_cometsrc_cons",
    ]
}
MODELS["random"] = None

for data_name, data_old in tqdm.tqdm(data_old_all):
    for repetitions, method in [
        (1, "cometsrc_avg"),
        (1, "cometsrc_var"),
        (1, "local_cometsrc_diffdisc"),
        (1, "cometsrc_diversity"),
        (1, "local_cometsrc_cons"),
        (100, "random"),
    ]:
        for _ in range(repetitions):
            data_new = subset2evaluate.select_subset.basic(data_old, method=method, load_model=MODELS[method])
            clu_new, cor_new = subset2evaluate.evaluate.eval_clucor(data_new, data_old, metric="human")
            points_y_cor[method].append(cor_new)
            points_y_clu[method].append(clu_new)


points_y_cor = {
    k: np.average(np.array(v), axis=0)
    for k, v in points_y_cor.items()
}
points_y_clu = {
    k: np.average(np.array(v), axis=0)
    for k, v in points_y_clu.items()
}

# %%
points_y_cor["cometsrc_cons"] = points_y_cor["local_cometsrc_cons"]
points_y_clu["cometsrc_cons"] = points_y_clu["local_cometsrc_cons"]
points_y_cor["cometsrc_diffdisc"] = points_y_cor["local_cometsrc_diffdisc"]
points_y_clu["cometsrc_diffdisc"] = points_y_clu["local_cometsrc_diffdisc"]

utils_fig.plot_subset_selection(
    [
        (utils.PROPS, points_y_cor["random"], f"Random {np.average(points_y_cor['random']):.1%}"),
        (utils.PROPS, points_y_cor["cometsrc_avg"], f"MetricAvg$^\\mathrm{{src}}$ {np.average(points_y_cor['cometsrc_avg']):.1%}"),
        (utils.PROPS, points_y_cor["cometsrc_var"], f"MetricVar$^\\mathrm{{src}}$ {np.average(points_y_cor['cometsrc_var']):.1%}"),
        (utils.PROPS, points_y_cor['cometsrc_cons'], f"MetricCons$^\\mathrm{{src}}$ {np.average(points_y_cor['cometsrc_cons']):.1%}"),
        (utils.PROPS, points_y_cor['cometsrc_diversity'], f"Diversity$^\\mathrm{{src}}$ {np.average(points_y_cor['cometsrc_diversity']):.1%}"),
        (utils.PROPS, points_y_cor['cometsrc_diffdisc'], f"DiffDisc$^\\mathrm{{src}}$ {np.average(points_y_cor['cometsrc_diffdisc']):.1%}"),
    ],
    colors=["#000000"] + utils_fig.COLORS,
    filename="14-main_sourcebased",
    height=1.9,
)
utils_fig.plot_subset_selection(
    [
        (utils.PROPS, points_y_clu["random"], f"Random {np.average(points_y_clu['random']):.2f}"),
        (utils.PROPS, points_y_clu["cometsrc_avg"], f"MetricAvg$^\\mathrm{{src}}$ {np.average(points_y_clu['cometsrc_avg']):.2f}"),
        (utils.PROPS, points_y_clu["cometsrc_var"], f"MetricVar$^\\mathrm{{src}}$ {np.average(points_y_clu['cometsrc_var']):.2f}"),
        (utils.PROPS, points_y_clu['cometsrc_cons'], f"MetricCons$^\\mathrm{{src}}$ {np.average(points_y_clu['cometsrc_cons']):.2f}"),
        (utils.PROPS, points_y_clu['cometsrc_diversity'], f"Diversity$^\\mathrm{{src}}$ {np.average(points_y_clu['cometsrc_diversity']):.2f}"),
        (utils.PROPS, points_y_clu['cometsrc_diffdisc'], f"DiffDisc$^\\mathrm{{src}}$ {np.average(points_y_clu['cometsrc_diffdisc']):.2f}"),
    ],
    colors=["#000000"] + utils_fig.COLORS,
    filename="14-main_sourcebased",
    height=1.9,
)
