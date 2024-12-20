# %%
import collections
import tqdm
import irt_mt_dev.utils as utils
import subset2evaluate.evaluate
import subset2evaluate.select_subset
import numpy as np
import os
import pickle
import irt_mt_dev.utils.fig

os.chdir("/home/vilda/irt-mt-dev")

data_old_all = list(utils.load_data_wmt_all(normalize=True).items())[:9]

# %%
data_train = collections.defaultdict(list)

# part I: train IRT on all data, distinguish where it came from
for data_name, data_old in tqdm.tqdm(data_old_all):
    # run multiple times and average item parameters
    for _ in range(5):
        _data, params = subset2evaluate.select_subset.run_select_subset(data_old, return_model=True, method="pyirt_diffdisc", model="4pl_score", metric="MetricX-23-c", epochs=1000, retry_on_error=True, enforce_positive_disc=True)
        data_train[data_name].append([
            {**line, "irt": line_irt}
            for line, line_irt in zip(data_old, params["items"])
        ])


pickle.dump(data_train, open("computed/tmp.pkl", "wb"))

# %%
data_train = pickle.load(open("computed/tmp.pkl", "rb"))

# %%
# part II
def average_irt_params(data_train):
    # the old data (e.g. line src) is the same everywhere
    data_new = [l for l in data_train[0]]
    irt_params = [collections.defaultdict(list) for _ in data_train]
    for i in range(len(data_train)):
        for data in data_train:
            for k, v in data[i]["irt"].items():
                irt_params[i][k].append(v)

    for i in range(len(data_train)):
        for k, v in irt_params[i].items():
            # TODO: try median?
            data_new[i]["irt"][k] = np.average(v)

    return data_new

data_train_avg = {
    k: average_irt_params(v)
    for k, v in data_train.items()
}

points_y_acc_all = collections.defaultdict(lambda: collections.defaultdict(list))
points_y_clu_all = collections.defaultdict(lambda: collections.defaultdict(list))

for data_name, data_old in tqdm.tqdm(data_old_all):
    # exclude test data from training
    data_train_flat = [
        line
        for data_local_name, data_local in data_train_avg.items()
        if data_name != data_local_name
        for line in data_local
    ]
    # NOTE: deeper MLP doesn't really work
    for _ in range(2):
        _data_new, model = subset2evaluate.select_subset.run_select_subset(data_old, method="premlp_irt_diffdisc", data_train=data_train_flat, return_model=True)
        for method in ["premlp_irt_diffdisc", "premlp_irt_disc", "premlp_irt_diff"]:
            data_new = subset2evaluate.select_subset.run_select_subset(data_old, method=method, data_train=data_train_flat, load_model=model)
            clu_new, acc_new = subset2evaluate.evaluate.run_evaluate_topk(data_old, data_new, metric="human")

            points_y_acc_all[method][data_name].append(acc_new)
            points_y_clu_all[method][data_name].append(clu_new)
        # train multiple times, there is some variance as well

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

irt_mt_dev.utils.fig.plot_subset_selection(
    [
        # (utils.PROPS, points_y_acc_all['pyirt_feas'], f"IRT feasability {np.average(points_y_acc_all['pyirt_feas']):.2%}"),
        (utils.PROPS, points_y_acc_all['premlp_irt_diff'], f"difficulty {np.average(points_y_acc_all['premlp_irt_diff']):.2%}"),
        (utils.PROPS, points_y_acc_all['premlp_irt_disc'], f"discriminability {np.average(points_y_acc_all['premlp_irt_disc']):.2%}"),
        (utils.PROPS, points_y_acc_all['premlp_irt_diffdisc'], f"diff.$\\times$disc. {np.average(points_y_acc_all['premlp_irt_diffdisc']):.2%}"),
        # (utils.PROPS, points_y_acc_all['pyirt_fic'], f"information {np.average(points_y_acc_all['pyirt_fic']):.2%}"),
    ],
    "23-irt_unseen_all",
)
irt_mt_dev.utils.fig.plot_subset_selection(
    [
        # (utils.PROPS, points_y_clu_all['pyirt_feas'], f"IRT feasability {np.average(points_y_clu_all['pyirt_feas']):.2f}"),
        (utils.PROPS, points_y_clu_all['premlp_irt_diff'], f"difficulty {np.average(points_y_clu_all['premlp_irt_diff']):.2f}"),
        (utils.PROPS, points_y_clu_all['premlp_irt_disc'], f"discriminability {np.average(points_y_clu_all['premlp_irt_disc']):.2f}"),
        (utils.PROPS, points_y_clu_all['premlp_irt_diffdisc'], f"diff.$\\times$disc. {np.average(points_y_clu_all['premlp_irt_diffdisc']):.2f}"),
        # (utils.PROPS, points_y_clu_all['pyirt_fic'], f"information {np.average(points_y_clu_all['pyirt_fic']):.2f}"),
    ],
    "23-irt_unseen_all",
)


# # %%
# data_new = subset2evaluate.select_subset.run_select_subset(data_old, method="random")
# clu_new, acc_new = subset2evaluate.evaluate.run_evaluate_topk(data_old, data_new, metric="human")
# print(np.average(clu_new))
# print(np.average(acc_new))