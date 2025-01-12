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

points_y_acc_all = collections.defaultdict(lambda: collections.defaultdict(list))
points_y_clu_all = collections.defaultdict(lambda: collections.defaultdict(list))

# cache models because that's where we lose a lot of time
MODELS = {
    method: subset2evaluate.select_subset.run_select_subset(data_old_all[0][1], method=method, return_model=True)[1]
    for method in ["precomet_diffdisc_direct", "precomet_diffdisc", "precomet_diversity"]
}

for data_name, data_old in tqdm.tqdm(data_old_all):
    for method, model in MODELS.items():
        data_new = subset2evaluate.select_subset.run_select_subset(data_old, method=method, load_model=model)
        clu_new, acc_new = subset2evaluate.evaluate.run_evaluate_topk(data_old, data_new, metric="human")
        points_y_acc_all[method][data_name].append(acc_new)
        points_y_clu_all[method][data_name].append(clu_new)

# %%
# average results
points_y_acc_all = {
    method: np.average(np.array(list(method_v.values())), axis=(0, 1))
    for method, method_v in points_y_acc_all.items()
}
points_y_clu_all = {
    method: np.average(np.array(list(method_v.values())), axis=(0, 1))
    for method, method_v in points_y_clu_all.items()
}

# %%

utils_fig.plot_subset_selection(
    [
        (utils.PROPS, points_y_acc_all['precomet_diversity'], f"PreCOMET$^\\mathrm{{div.}}$ {np.average(points_y_acc_all['precomet_diversity']):.1%}"),
        (utils.PROPS, points_y_acc_all['precomet_diffdisc'], f"PreCOMET$^\\mathrm{{diff.\\hspace{{-0.5}}×diff.}}$ {np.average(points_y_acc_all['precomet_diffdisc']):.1%}"),
        (utils.PROPS, points_y_acc_all['precomet_diffdisc_direct'], f"PreCOMET$^\\mathrm{{diff.\\hspace{{-0.5}}×diff}}$ $\\hspace{{-3.3}}_\mathrm{{direct}}\\hspace{{1.8}}$ {np.average(points_y_acc_all['precomet_diffdisc_direct']):.1%}"),
    ],
    "23-main_sourcebased_other",
)
utils_fig.plot_subset_selection(
    [
        (utils.PROPS, points_y_clu_all['precomet_diversity'], f"PreCOMET$^\\mathrm{{div.}}$ {np.average(points_y_clu_all['precomet_diversity']):.2f}"),
        (utils.PROPS, points_y_clu_all['precomet_diffdisc'], f"PreCOMET$^\\mathrm{{diff.\\times diff.}}$ {np.average(points_y_clu_all['precomet_diffdisc']):.2f}"),
        (utils.PROPS, points_y_clu_all['precomet_diffdisc_direct'], f"PreCOMET$^\\mathrm{{diff.\\hspace{{-0.5}}×diff}}$ $\\hspace{{-3.3}}_\mathrm{{direct}}\\hspace{{1.8}}$ {np.average(points_y_clu_all['precomet_diffdisc_direct']):.2f}"),
    ],
    "23-main_sourcebased_other",
)