# %%
import subset2evaluate
import subset2evaluate.evaluate
import subset2evaluate.select_subset
import subset2evaluate.utils
import numpy as np
import multiprocessing


def process_data_old(data_old, seed, target, kwargs):
    k = int(0.25*len(data_old))
    data_new = subset2evaluate.select_subset.basic(data_old, method="random", seed=seed)
    acc_rand = subset2evaluate.evaluate.eval_subset_accuracy(data_new[:k], data_old, metric=target)

    acc_better = None
    k = 0
    while acc_better is None or acc_better < acc_rand:
        k += 1
        data_new = subset2evaluate.select_subset.basic(data_old, **kwargs)
        acc_better = subset2evaluate.evaluate.eval_subset_accuracy(data_new[:k], data_old, metric=target)

    return k/(len(data_old)*0.25)

# %% 
data_old_all = list(subset2evaluate.utils.load_data("wmt23/all").items())[:9]
with multiprocessing.Pool(20) as pool:
    req_k = pool.starmap(
        process_data_old,
        [
            (data_old, seed, "human", dict(method="metric_var", metric="MetricX-23"))
            for seed in range(5) for _, data_old in data_old_all
        ])

print(f"{np.average(req_k):.1%}")

# %%

data_old = subset2evaluate.utils.load_data("summeval")
with multiprocessing.Pool(5) as pool:
    req_k = pool.starmap(
        process_data_old,
        [
            (data_old, seed, "human_sum", dict(method="metric_var", metric="coverage"))
            for seed in range(5)
        ])

print(f"{np.average(req_k):.1%}")
