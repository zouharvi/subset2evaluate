# %%

import subset2evaluate
import subset2evaluate.utils
import subset2evaluate.evaluate
import subset2evaluate.select_subset
import numpy as np
import collections

data_old = subset2evaluate.utils.load_data_summeval(normalize=True, load_extra=True)
PROPS = np.linspace(0.25, 0.75, 10)

# %%
METRIC_TARGET = ["gpt_relevance", "gpt_coherence", "gpt_consistency", "gpt_fluency", "gpt_sum"]
spa_precomputed = {
    m: subset2evaluate.evaluate.precompute_spa_randnorm(data_old, metric=m, props=PROPS)
    for m in METRIC_TARGET
}

# %%

# parity
for repetitions, method_kwargs in [
    (1, dict(method="metric_avg", metric="supert")),
    (1, dict(method="metric_var", metric="supert")),
    (1, dict(method="metric_cons", metric="supert")),
    (1, dict(method="diversity", metric="LM")),
    (5, dict(method="pyirt_diffdisc", retry_on_error=True)),
]:
    spa_local = []
    # for metric_target in ["gpt_relevance", "gpt_coherence", "gpt_consistency", "gpt_fluency", "gpt_sum"]:
    for _ in range(repetitions):
        for metric_target in METRIC_TARGET:
            par_spa = subset2evaluate.evaluate.eval_spa_par_randnorm(
                subset2evaluate.select_subset.basic(data_old, **method_kwargs),
                data_old,
                metric=metric_target,
                props=PROPS,
                spa_precomputed=spa_precomputed[metric_target],
            )
            spa_local.append(par_spa)
    print(method_kwargs["method"], f"{np.average(spa_local):.1%}")


# %%
spa_all = collections.defaultdict(list)
for repetitions, method_kwargs in [
    (100, dict(method="random")),
    (1, dict(method="metric_avg", metric="supert")),
    (1, dict(method="metric_var", metric="supert")),
    (1, dict(method="metric_cons", metric="supert")),
    (1, dict(method="diversity", metric="LM")),
    (5, dict(method="pyirt_diffdisc", metric="supert", retry_on_error=True)),
]:
    for _ in range(repetitions):
        data_new = subset2evaluate.select_subset.basic(data_old, **method_kwargs)
        for metric_target in ["gpt_relevance", "gpt_coherence", "gpt_consistency", "gpt_fluency", "gpt_sum"]:
            spa_new = subset2evaluate.evaluate.eval_spa(data_new, data_old, metric=metric_target, props=PROPS)
            spa_all[method_kwargs['method']].append(spa_new)
    print(method_kwargs["method"], f"{np.average(spa_all[method_kwargs['method']]):.1%}")


# %%
# find best metric: supert
_ = subset2evaluate.evaluate.eval_metrics_correlations(data_old, metric_target="gpt_sum", display=True)