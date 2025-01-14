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

points_y_acc = collections.defaultdict(lambda: collections.defaultdict(list))
points_y_clu = collections.defaultdict(lambda: collections.defaultdict(list))

# cache models because that's where we lose a lot of time
MODELS = {
    method: subset2evaluate.select_subset.run_select_subset(data_old_all[0][1], method=method, return_model=True)[1]
    for method in ["precomet_diffdisc", "precomet_diversity"]
}
MODELS["random"] = None

for data_name, data_old in tqdm.tqdm(data_old_all):
    for repetitions, method in [(1, "precomet_diffdisc"), (1, "precomet_diversity"), (100, "random")]:
        for _ in range(repetitions):
            data_new = subset2evaluate.select_subset.run_select_subset(data_old, method=method, load_model=MODELS[method])
            clu_new, acc_new = subset2evaluate.evaluate.eval_cluacc(data_new, data_old, metric="human")
            points_y_acc[method].append(acc_new)
            points_y_clu[method].append(clu_new)


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
    [
        (utils.PROPS, points_y_acc["random"], f"Random {np.average(points_y_acc['random']):.1%}"),
        (utils.PROPS, points_y_acc['precomet_diversity'], f"PreCOMET$^\\mathrm{{div.}}$ {np.average(points_y_acc['precomet_diversity']):.1%}"),
        (utils.PROPS, points_y_acc['precomet_diffdisc'], f"PreCOMET$^\\mathrm{{diff.\\hspace{{-0.5}}×diff.}}$ {np.average(points_y_acc['precomet_diffdisc']):.1%}"),
        # (utils.PROPS, points_y_acc['precomet_diffdisc_direct'], f"PreCOMET$^\\mathrm{{diff.\\hspace{{-0.5}}×diff}}$ $\\hspace{{-3.3}}_\\mathrm{{direct}}\\hspace{{1.8}}$ {np.average(points_y_acc['precomet_diffdisc_direct']):.1%}"),
    ],
    colors=["black"] + utils_fig.COLORS,
    filename="23-main_sourcebased_other",
)
utils_fig.plot_subset_selection(
    [
        (utils.PROPS, points_y_clu["random"], f"Random {np.average(points_y_clu['random']):.2f}"),
        (utils.PROPS, points_y_clu['precomet_diversity'], f"PreCOMET$^\\mathrm{{div.}}$ {np.average(points_y_clu['precomet_diversity']):.2f}"),
        (utils.PROPS, points_y_clu['precomet_diffdisc'], f"PreCOMET$^\\mathrm{{diff.\\times diff.}}$ {np.average(points_y_clu['precomet_diffdisc']):.2f}"),
        # (utils.PROPS, points_y_clu['precomet_diffdisc_direct'], f"PreCOMET$^\\mathrm{{diff.\\hspace{{-0.5}}×diff}}$ $\\hspace{{-3.3}}_\\mathrm{{direct}}\\hspace{{1.8}}$ {np.average(points_y_clu['precomet_diffdisc_direct']):.2f}"),
    ],
    colors=["black"] + utils_fig.COLORS,
    filename="23-main_sourcebased_other",
)
