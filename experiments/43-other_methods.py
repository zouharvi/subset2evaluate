# %%

import json
import subset2evaluate
import subset2evaluate.evaluate
import subset2evaluate.select_subset
import subset2evaluate.utils
import subset2evaluate.methods
import utils_fig
import tqdm
import numpy as np

data_old_all = list(subset2evaluate.utils.load_data_wmt_test().values())

# %%
# "standard" subset selection methods
for method_kwargs in [
    dict(method="metric_avg", metric="human"),
    dict(method="metric_var", metric="human"),
    dict(method="metric_cons", metric="human"),
    dict(method="diversity", metric="bleu"),
    dict(method="diversity", metric="chrf"),
    dict(method="diversity", metric="unigram"),
]:
    load_model = None
    spa_all = []
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
for method_kwargs in [
    # dict(method="kmeans", features="tgt_0"),
    # dict(method="kmeans", features="src"),
    dict(method="bruteforce", metric="human", simulations=1000),
    dict(method="bruteforce", metric="MetricX-23-c", simulations=1000),
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

# %%
utils_fig.plot_subset_selection(
    points=[
        (subset2evaluate.utils.PROPS, np.average(spa_all, axis=0), f"k-means {np.average(spa_all):.2f}"),
    ],
    measure="spa",
    filename="43-kmeans",
    colors=["#000000"] + utils_fig.COLORS,
    height=1.5,
)