# %%

import json
import subset2evaluate
import subset2evaluate.evaluate
import subset2evaluate.select_subset
import subset2evaluate.utils
import subset2evaluate.methods
import tqdm
import numpy as np

data_old_all = list(subset2evaluate.utils.load_data_wmt_test().values())

# %%
# "standard" subset selection methods
for repetitions, method_kwargs in [
    (1, dict(method="metric_avg", metric="human")),
    (1, dict(method="metric_var", metric="human")),
    (1, dict(method="metric_cons", metric="human")),
    (1, dict(method="diversity", metric="bleu")),
    (1, dict(method="diversity", metric="chrf")),
    (1, dict(method="diversity", metric="unigram")),
    (5, dict(method="pyirt_diff", metric="MetricX-23-c", retry_on_error=True)),
    (5, dict(method="pyirt_disc", metric="MetricX-23-c", retry_on_error=True)),
    (5, dict(method="pyirt_feas", metric="MetricX-23-c", retry_on_error=True)),
    (5, dict(method="pyirt_fic", metric="MetricX-23-c", retry_on_error=True)),
    (1, dict(method="sentinel_src_da")),
    (1, dict(method="sentinel_src_mqm")),
    (1, dict(method="metric_var", metric="MetricX-23-c")),
    (10, dict(
        method="combinator",
        operation="avg",
        normalize="rank",
        methods=[
            (3, dict(method="metric_var", metric="MetricX-23-c")),
            (1, dict(method="random")),
        ]
    )),
    (1, dict(
        method="combinator",
        operation="avg",
        normalize="rank",
        methods=[
            (1, dict(method="metric_var", metric="MetricX-23-c")),
            (1, dict(method="metric_avg", metric="MetricX-23-c")),
        ]
    )),
    (1, dict(
        method="combinator",
        operation="avg",
        normalize="rank",
        methods=[
            (1, dict(method="metric_avg", metric="MetricX-23-c")),
            (1, dict(method="metric_avg", metric="XCOMET-XXL")),
        ]
    )),
    (1, dict(
        method="combinator",
        operation="avg",
        normalize="rank",
        methods=[
            (1, dict(method="metric_var", metric="MetricX-23-c")),
            (1, dict(method="metric_var", metric="XCOMET-XXL")),
        ]
    )),
    (1, dict(
        method="combinator",
        operation="avg",
        normalize="rank",
        methods=[
            (1, dict(method="metric_cons", metric="MetricX-23-c")),
            (1, dict(method="diversity", metric="lm")),
            (1, dict(method="random")),
        ]
    )),
]:
    load_model = None
    spa_all = []
    for _ in range(repetitions):
        for data_old in tqdm.tqdm(data_old_all):
            spa_new = subset2evaluate.evaluate.eval_spa(
                subset2evaluate.select_subset.basic(data_old, **method_kwargs),
                data_old,
                metric="human",
                props=subset2evaluate.utils.PROPS,
            )
            spa_all.append(spa_new)
    print(f"{json.dumps(method_kwargs)}: {np.average(spa_all):.1%}")

# %%

# %%
# methods that have to be run with a specific budget

for method_kwargs in [
    dict(method="diffuse"),
    # dict(method="kmeans", features="tgt_0"),
    # dict(method="kmeans", features="src"),
    # dict(method="bruteforce", metric="human", simulations=1), # this is random!
    # dict(method="bruteforce", metric="human", simulations=10),
    # dict(method="bruteforce", metric="human", simulations=100),
    # dict(method="bruteforce", metric="human", simulations=1000),
    # dict(method="bruteforce", metric="MetricX-23-c", simulations=10),
    # dict(method="bruteforce", metric="MetricX-23-c", simulations=100),
    # dict(method="bruteforce", metric="MetricX-23-c", simulations=1000),
    # dict(method="bruteforce_greedy", metric="human", simulations=10, stepsize=10),
    # dict(method="bruteforce_greedy", metric="human", simulations=100, stepsize=10),
    # dict(method="bruteforce_greedy", metric="MetricX-23-c", simulations=10, stepsize=10),
    # dict(method="bruteforce_greedy", metric="MetricX-23-c", simulations=100, stepsize=10),
]:
    load_model = None
    spa_all = []
    for data_old in tqdm.tqdm(data_old_all):
        spa_local = []
        for prop in subset2evaluate.utils.PROPS:
            B = int(len(data_old) * prop)
            data_new, load_model = subset2evaluate.select_subset.basic(data_old, **method_kwargs, budget=B, load_model=load_model, return_model=True)
            data_new = data_new[:B]
            spa_new = subset2evaluate.evaluate.eval_subset_spa(data_new, data_old, metric="human")
            spa_local.append(spa_new)
        spa_all.append(spa_local)
    print(f"{json.dumps(method_kwargs)}: {np.average(spa_all):.1%}")

# %%

# special handling for k-means with supersampling

for method_kwargs in [
    dict(supersample=True, features="tgt_0"),
    dict(supersample=True, features="src"),
]:
    load_model = None
    spa_all = []
    for data_old in tqdm.tqdm(data_old_all):
        spa_local = []
        for prop in subset2evaluate.utils.PROPS:
            B = int(len(data_old) * prop)
            data_new, load_model = subset2evaluate.methods.kmeans_supersample(data_old, **method_kwargs, budget=B, load_model=load_model, return_model=True)
            spa_new = subset2evaluate.evaluate.eval_subset_spa(data_new, data_old, metric="human")
            spa_local.append(spa_new)

        spa_all.append(spa_local)
    print(f"{json.dumps(method_kwargs)}: {np.average(spa_all):.1%}")
