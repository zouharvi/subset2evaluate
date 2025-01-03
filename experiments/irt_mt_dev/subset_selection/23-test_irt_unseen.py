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
        _data, params = subset2evaluate.select_subset.run_select_subset(data_old, return_model=True, method="pyirt_diffdisc", model="4pl_score", metric="human", epochs=1000, retry_on_error=True, enforce_positive_disc=True)
        data_train[data_name].append([
            {**line, "irt": line_irt}
            for line, line_irt in zip(data_old, params["items"])
        ])


pickle.dump(data_train, open("computed/test_irt_unseen.pkl", "wb"))

# %%
# part II

data_train = pickle.load(open("computed/test_irt_unseen.pkl", "rb"))
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
    for _ in range(2):
        data_new = subset2evaluate.select_subset.run_select_subset(data_old, method="premlp_irt_diffdisc", data_train=data_train_flat)
        clu_new, acc_new = subset2evaluate.evaluate.run_evaluate_topk(data_old, data_new, metric="human")
        points_y_acc_all["premlp_irt_diffdisc"][data_name].append(acc_new)
        points_y_clu_all["premlp_irt_diffdisc"][data_name].append(clu_new)
        # train multiple times, there is some variance as well
    for _ in range(1):
        data_new = subset2evaluate.select_subset.run_select_subset(data_old, method="precomet_div")
        clu_new, acc_new = subset2evaluate.evaluate.run_evaluate_topk(data_old, data_new, metric="human")

        points_y_acc_all["precomet_div"][data_name].append(acc_new)
        points_y_clu_all["precomet_div"][data_name].append(clu_new)

# %%
points_y_acc_all_backup = points_y_acc_all
points_y_clu_all_backup = points_y_clu_all

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


# %%

irt_mt_dev.utils.fig.plot_subset_selection(
    [
        (utils.PROPS, points_y_acc_all['premlp_irt_diffdisc'], f"diff.$\\times$disc. {np.average(points_y_acc_all['premlp_irt_diffdisc']):.2%}"),
        (utils.PROPS, points_y_acc_all['precomet_div'], f"PreCOMET div. {np.average(points_y_acc_all['precomet_div']):.2%}"),
    ],
    "23-irt_unseen_all",
)
irt_mt_dev.utils.fig.plot_subset_selection(
    [
        (utils.PROPS, points_y_clu_all['premlp_irt_diffdisc'], f"diff.$\\times$disc. {np.average(points_y_clu_all['premlp_irt_diffdisc']):.2f}"),
        (utils.PROPS, points_y_clu_all['precomet_div'], f"PreCOMET div. {np.average(points_y_clu_all['precomet_div']):.2f}"),
    ],
    "23-irt_unseen_all",
)