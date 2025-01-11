# %%

import subset2evaluate
import subset2evaluate.utils as utils
import numpy as np

data_old = utils.load_data_summeval(normalize=True)

# %%
for target in ["human_relevance", "human_coherence", "human_consistency", "human_fluency"]:
    acc_all = []
    clu_all = []
    for _ in range(100):
        data_new = subset2evaluate.select_subset.run_select_subset(data_old, method="random")
        clu_new, acc_new = subset2evaluate.evaluate.run_evaluate_topk(data_old, data_new, metric=target)
        acc_all.append(acc_new)
        clu_all.append(clu_new)
    print(target, f"ACC: {np.average(acc_all):.2%} | CLU: {np.average(clu_all):.2f}")

# %%

for target, metric in [
    ("human_relevance", "chrf"),
    ("human_coherence", "density"),
    ("human_consistency", "coverage")
    ("human_fluency", "coverage"),
]:
    data_new = subset2evaluate.select_subset.run_select_subset(data_old, method="avg", metric=metric)
    clu_new, acc_new = subset2evaluate.evaluate.run_evaluate_topk(data_old, data_new, metric=target)
    print(target, f"ACC: {np.average(acc_new):.2%} | CLU: {np.average(clu_new):.2f}")

# %%

for target, metric in [
    ("human_relevance", "chrf"),
    ("human_coherence", "density"),
    ("human_consistency", "coverage")
    ("human_fluency", "coverage"),
]:
    data_new = subset2evaluate.select_subset.run_select_subset(data_old, method="var", metric=metric)
    clu_new, acc_new = subset2evaluate.evaluate.run_evaluate_topk(data_old, data_new, metric=target)
    print(target, f"ACC: {np.average(acc_new):.2%} | CLU: {np.average(clu_new):.2f}")

# %%

for target, metric in [
    ("human_relevance", "chrf"),
    ("human_coherence", "density"),
    ("human_consistency", "coverage")
    ("human_fluency", "coverage"),
]:
    acc_all = []
    clu_all = []
    for _ in range(5):
        data_new = subset2evaluate.select_subset.run_select_subset(data_old, method="pyirt_diffdisc", metric=metric, retry_on_error=True)
        clu_new, acc_new = subset2evaluate.evaluate.run_evaluate_topk(data_old, data_new, metric=target)
        acc_all.append(acc_new)
        clu_all.append(clu_new)
    print(target, f"ACC: {np.average(acc_all):.2%} | CLU: {np.average(clu_all):.2f}")


# %%
for target in ["human_relevance", "human_coherence", "human_fluency", "human_consistency"]:
    data_new = subset2evaluate.select_subset.run_select_subset(data_old, method="diversity")
    clu_new, acc_new = subset2evaluate.evaluate.run_evaluate_topk(data_old, data_new, metric=target)
    print(target, f"ACC: {np.average(acc_new):.2%} | CLU: {np.average(clu_new):.2f}")