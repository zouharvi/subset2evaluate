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
        metric="human_sum",
    )
    print(method_kwargs["method"], f"ACC: {par_acc:.1%} | CLU: {par_clu:.1%}")

# %%
# ["human_relevance", "human_coherence", "human_consistency", "human_fluency", "human_sum", "human_mul"]


for repetitions, method_kwargs in [
    (100, dict(method="random")),
    (1, dict(method="metric_alignment", metric="supert")),
    (1, dict(method="metric_avg", metric="supert")),
    (1, dict(method="metric_var", metric="supert")),
    (1, dict(method="diversity_bleu")),
    (5, dict(method="pyirt_diffdisc", metric="supert", retry_on_error=True)),
]:
    cor_all = []
    clu_all = []
    for _ in range(repetitions):
        data_new = subset2evaluate.select_subset.basic(data_old, **method_kwargs)
        clu_new, cor_new = subset2evaluate.evaluate.eval_clucor(data_new, data_old, metric="human_sum")
        cor_all.append(cor_new)
        clu_all.append(clu_new)
    print(method_kwargs["method"], f"COR: {np.average(cor_all):.1%} | CLU: {np.average(clu_all):.2f}")
