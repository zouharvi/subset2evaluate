# %%

import tqdm
import subset2evaluate.utils as utils
import numpy as np
import os
import subset2evaluate.evaluate
import subset2evaluate.select_subset
import collections
import copy
import itertools
import sacrebleu

os.chdir("/home/vilda/irt-mt-dev")

def utility_metricx_avg(item):
    return -np.average(
        [sys_v["MetricX-23-c"] for sys_v in item["scores"].values()]
    )

def utility_metricx_var(item):
    return np.var(
        [sys_v["MetricX-23-c"] for sys_v in item["scores"].values()]
    )

def utility_irt_diffdisc(item, data_irt):
    item_irt = data_irt["items"][item["i"]]
    return item_irt["diff"]*item_irt["disc"]

metric_bleu = sacrebleu.metrics.BLEU(effective_order=True)
def utility_diversity(line):
    return -np.average([
        metric_bleu.sentence_score(
            text_a,
            [text_b],
        ).score
        for text_a, text_b in itertools.product(line["tgt"].values(), line["tgt"].values())
    ])


acc_all_all = collections.defaultdict(list)
clu_all_all = collections.defaultdict(list)

data_old_all = list(utils.load_data_wmt_all(normalize=True).values())[:9]
for data_old in tqdm.tqdm(data_old_all):
    _, irt_params = subset2evaluate.select_subset.run_select_subset(data_old, method="pyirt_diffdisc", metric="MetricX-23-c", epochs=1000, retry_on_error=True, return_model=True)

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
    for beta in [0, 1, 3]:
        # random is independent of beta but let's average it!
        data_random = subset2evaluate.select_subset.run_select_subset(data_old, method="random")
        clu_new, acc_new = subset2evaluate.evaluate.run_evaluate_top_timebudget(data_old, data_random, metric="human")
        acc_all["random"].append(np.average(acc_new))
        clu_all["random"].append(np.average(clu_new))

        clu_new, acc_new = beta_searcher_evaluate(utility_fn=utility_metricx_avg, beta=beta)
        acc_all["metricavg"].append(np.average(acc_new))
        clu_all["metricavg"].append(np.average(clu_new))

        clu_new, acc_new = beta_searcher_evaluate(utility_fn=utility_metricx_var, beta=beta)
        acc_all["metricvar"].append(np.average(acc_new))
        clu_all["metricvar"].append(np.average(clu_new))

        clu_new, acc_new = beta_searcher_evaluate(utility_fn=utility_diversity, beta=beta)
        acc_all["diversity"].append(np.average(acc_new))
        clu_all["diversity"].append(np.average(clu_new))

        clu_new, acc_new = beta_searcher_evaluate(utility_fn=lambda x: utility_irt_diffdisc(x, irt_params), beta=beta)
        acc_all["irt_diffdisc"].append(np.average(acc_new))
        clu_all["irt_diffdisc"].append(np.average(clu_new))

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
printrow(np.average(acc_all_all["diversity"], axis=(0,)))
printrow(np.average(acc_all_all["irt_diffdisc"], axis=(0,)))

def printrow(row):
    print(" & ".join([f"{x:.2f}" for x in row]) + r" \\")
print(f"{np.average(clu_all_all['random']):.2f} \\\\")
printrow(np.average(clu_all_all["metricavg"], axis=(0,)))
printrow(np.average(clu_all_all["metricvar"], axis=(0,)))
printrow(np.average(clu_all_all["diversity"], axis=(0,)))
printrow(np.average(clu_all_all["irt_diffdisc"], axis=(0,)))
