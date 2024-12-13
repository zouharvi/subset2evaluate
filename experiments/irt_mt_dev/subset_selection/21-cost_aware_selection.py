# %%


import irt_mt_dev.utils as utils
import irt_mt_dev.utils.fig
import numpy as np
import os
import subset2evaluate.evaluate
import subset2evaluate.select_subset
import tqdm
import collections
import copy

os.chdir("/home/vilda/irt-mt-dev")

def utility_metricx_avg(item):
    return -np.average(
        [sys_v["MetricX-23-c"] for sys_v in item["scores"].values()]
    )

def utility_metricx_var(item):
    return np.var(
        [sys_v["MetricX-23-c"] for sys_v in item["scores"].values()]
    )

def utility_irt_fic(item, data_irt):
    # aggregared fisher information content
    item = data_irt["items"][item["i"]]

    information = 0
    for theta in data_irt["systems"].values():
        prob = utils.pred_irt(
            theta,
            item
        )
        information += prob*(1-prob)*(item["disc"]**2)
    return information


acc_all_all = collections.defaultdict(list)
clu_all_all = collections.defaultdict(list)

data_old_all = list(utils.load_data_wmt_all(normalize=True).values())[:9]
for data_old in data_old_all:
    _, irt_params = subset2evaluate.select_subset.run_select_subset(data_old, method="pyirt_fic", metric="MetricX-23-c", epochs=1000, retry_on_error=True, return_model=True)

    def beta_searcher_evaluate(utility_fn, beta):
        data_utility = np.array([utility_fn(item) for item in data_old])
        # z-normalize
        data_utility = (data_utility - np.mean(data_utility)) / np.std(data_utility)
        data_time = np.array([item["time"] for item in data_old])
        data_utility = data_utility - beta * data_time
        data_new = copy.deepcopy(data_old)
        data_new_util = list(zip(data_new, data_utility))
        data_new_util.sort(key=lambda x: x[1], reverse=True)
        data_new = [x[0] for x in data_new_util]
        return subset2evaluate.evaluate.run_evaluate_top_timebudget(data_old, data_new, metric="human")

    acc_all = collections.defaultdict(list)
    clu_all = collections.defaultdict(list)
    for beta in [0, 0.5, 1, float("inf")]:
        # random is independent of beta but let's average it!
        data_random = subset2evaluate.select_subset.run_select_subset(data_old, method="random")
        (_, clu_new), acc_new = subset2evaluate.evaluate.run_evaluate_top_timebudget(data_old, data_random, metric="human")
        acc_all["random"].append(np.average(acc_new))
        clu_all["random"].append(np.average(clu_new))

        (_, clu_new), acc_new = beta_searcher_evaluate(utility_fn=utility_metricx_avg, beta=beta)
        acc_all["metricavg"].append(np.average(acc_new))
        clu_all["metricavg"].append(np.average(clu_new))


        (_, clu_new), acc_new = beta_searcher_evaluate(utility_fn=utility_metricx_var, beta=beta)
        acc_all["metricvar"].append(np.average(acc_new))
        clu_all["metricvar"].append(np.average(clu_new))

        (_, clu_new), acc_new = beta_searcher_evaluate(utility_fn=lambda x: utility_irt_fic(x, irt_params), beta=beta)
        acc_all["irt_fic"].append(np.average(acc_new))
        clu_all["irt_fic"].append(np.average(clu_new))

    for key, value in acc_all.items():
        acc_all_all[key].append(value)
    for key, value in clu_all.items():
        clu_all_all[key].append(value)


# %%
def printrow(row):
    print(" & ".join([f"{x:.2%}".replace("%", r"\%") for x in row]) + r" \\")
print(f"{np.average(acc_all_all['random']):.2%} \\\\")
printrow(np.average(acc_all_all["metricavg"], axis=(0,)))
printrow(np.average(acc_all_all["metricvar"], axis=(0,)))
printrow(np.average(acc_all_all["irt_fic"], axis=(0,)))

def printrow(row):
    print(" & ".join([f"{x:.2f}" for x in row]) + r" \\")
print(f"{np.average(clu_all_all['random']):.2f} \\\\")
printrow(np.average(clu_all_all["metricavg"], axis=(0,)))
printrow(np.average(clu_all_all["metricvar"], axis=(0,)))
printrow(np.average(clu_all_all["irt_fic"], axis=(0,)))