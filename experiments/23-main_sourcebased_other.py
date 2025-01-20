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
    for method in ["local_precomet_diffdisc", "precomet_diversity", "local_precomet_ali"]
}
MODELS["random"] = None

for data_name, data_old in tqdm.tqdm(data_old_all):
    for repetitions, method in [(1, "local_precomet_diffdisc"), (1, "precomet_diversity"), (1, "local_precomet_ali"), (100, "random")]:
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
points_y_cor["precomet_ali"] = points_y_cor["local_precomet_ali"]
points_y_clu["precomet_ali"] = points_y_clu["local_precomet_ali"]
points_y_cor["precomet_diffdisc"] = points_y_cor["local_precomet_diffdisc"]
points_y_clu["precomet_diffdisc"] = points_y_clu["local_precomet_diffdisc"]

utils_fig.plot_subset_selection(
    [
        (utils.PROPS, points_y_cor["random"], f"Random {np.average(points_y_cor['random']):.1%}"),
        (utils.PROPS, points_y_cor['precomet_diversity'], f"Diversity$^\\mathrm{{src}}$ {np.average(points_y_cor['precomet_diversity']):.1%}"),
        (utils.PROPS, points_y_cor['precomet_diffdisc'], f"Diff.$^\\mathrm{{src}}$×Disc.$^\\mathrm{{src}}$ {np.average(points_y_cor['precomet_diffdisc']):.1%}"),
        (utils.PROPS, points_y_cor['precomet_ali'], f"MetricX align.$^\\mathrm{{src}}$ {np.average(points_y_cor['precomet_ali']):.1%}"),
    ],
    colors=["black"] + utils_fig.COLORS,
    filename="23-main_sourcebased_other",
)
utils_fig.plot_subset_selection(
    [
        (utils.PROPS, points_y_clu["random"], f"Random {np.average(points_y_clu['random']):.2f}"),
        (utils.PROPS, points_y_clu['precomet_diversity'], f"Diversity$^\\mathrm{{src}}$ {np.average(points_y_clu['precomet_diversity']):.2f}"),
        (utils.PROPS, points_y_clu['precomet_diffdisc'], f"Diff.$^\\mathrm{{src}}$×Disc.$^\\mathrm{{src}}$ {np.average(points_y_clu['precomet_diffdisc']):.2f}"),
        (utils.PROPS, points_y_clu['precomet_ali'], f"MetricX align.$^\\mathrm{{src}}$ {np.average(points_y_clu['precomet_ali']):.2f}"),
    ],
    colors=["black"] + utils_fig.COLORS,
    filename="23-main_sourcebased_other",
)
