# %%

import subset2evaluate
import subset2evaluate.evaluate
import numpy as np
# 
data_old = subset2evaluate.utils.load_data_summeval(normalize=True)

# %%

for method_kwargs in [
    dict(method="metric_var", metric="supert"),
    dict(method="metric_avg", metric="supert"),
    dict(method="pyirt_diffdisc", metric="supert"),
    dict(method="diversity_bleu"),
]:
    par_clu, par_acc = subset2evaluate.evaluate.eval_clucor_randnorm(
        subset2evaluate.select_subset.basic(data_old, **method_kwargs),
        data_old,
        metric="human_mul",
    )
    print(method_kwargs["method"], f"ACC: {par_acc:.1%} | CLU: {par_clu:.1%}")


# %%
for target in ["human_relevance", "human_coherence", "human_consistency", "human_fluency", "human_sum", "human_mul"]:
    acc_all = []
    clu_all = []
    for _ in range(100):
        data_new = subset2evaluate.select_subset.basic(data_old, method="random")
        clu_new, cor_new = subset2evaluate.evaluate.eval_clucor(data_new, data_old, metric=target)
        acc_all.append(cor_new)
        clu_all.append(clu_new)
    print(target, f"ACC: {np.average(acc_all):.1%} | CLU: {np.average(clu_all):.2f}")

# %%

for target, metric in [
    ("human_relevance", "chrf"),
    ("human_coherence", "density"),
    ("human_consistency", "coverage")
    ("human_fluency", "coverage"),
    ("human_sum", "supert"),
    ("human_mul", "supert"),
]:
    data_new = subset2evaluate.select_subset.basic(data_old, method="metric_avg", metric=metric)
    clu_new, cor_new = subset2evaluate.evaluate.eval_clucor(data_new, data_old, metric=target)
    print(target, f"ACC: {np.average(cor_new):.1%} | CLU: {np.average(clu_new):.2f}")

# %%

for target, metric in [
    ("human_relevance", "chrf"),
    ("human_coherence", "density"),
    ("human_consistency", "coverage")
    ("human_fluency", "coverage"),
    ("human_sum", "supert"),
    ("human_mul", "supert"),
]:
    data_new = subset2evaluate.select_subset.basic(data_old, method="metric_var", metric=metric)
    clu_new, cor_new = subset2evaluate.evaluate.eval_clucor(data_new, data_old, metric=target)
    print(target, f"ACC: {np.average(cor_new):.1%} | CLU: {np.average(clu_new):.2f}")

# %%

for target, metric in [
    ("human_relevance", "chrf"),
    ("human_coherence", "density"),
    ("human_consistency", "coverage")
    ("human_fluency", "coverage"),
    ("human_sum", "supert"),
    ("human_mul", "supert"),
]:
    acc_all = []
    clu_all = []
    for _ in range(5):
        data_new = subset2evaluate.select_subset.basic(data_old, method="pyirt_diffdisc", metric=metric, retry_on_error=True)
        clu_new, cor_new = subset2evaluate.evaluate.eval_clucor(data_new, data_old, metric=target)
        acc_all.append(cor_new)
        clu_all.append(clu_new)
    print(target, f"ACC: {np.average(acc_all):.1%} | CLU: {np.average(clu_all):.2f}")


# %%
for target in ["human_relevance", "human_coherence", "human_fluency", "human_consistency", "human_sum", "human_mul"]:
    data_new = subset2evaluate.select_subset.basic(data_old, method="diversity_bleu")
    clu_new, cor_new = subset2evaluate.evaluate.eval_clucor(data_new, data_old, metric=target)
    print(target, f"ACC: {np.average(cor_new):.1%} | CLU: {np.average(clu_new):.2f}")
