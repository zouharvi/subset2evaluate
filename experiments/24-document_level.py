# %%
import collections
import subset2evaluate.utils as utils
import subset2evaluate.evaluate
import subset2evaluate.select_subset
import numpy as np
import tqdm

data_old_all = list(utils.load_data_wmt_all(normalize=True).values())[:9]


# %%
cor_new_all = collections.defaultdict(list)
clu_new_all = collections.defaultdict(list)

for data_old in tqdm.tqdm(data_old_all):
    def evaluate_aggregate_second(data_scored):
        data_old_aggregated = collections.defaultdict(list)
        for line in data_scored:
            data_old_aggregated[line["doc"]].append(line)

        data_old_aggregated = [
            {
                "doc": doc,
                "i": [line["i"] for line in lines],
                # average the utilities
                "subset2evaluate_utility": np.average([line["subset2evaluate_utility"] for line in lines])
            }
            for doc, lines in data_old_aggregated.items()
        ]
        data_old_aggregated.sort(key=lambda x: x["subset2evaluate_utility"], reverse=True)
        data_new_flat = [
            data_old[i]
            for doc in data_old_aggregated
            for i in doc["i"]
        ]
        clu_new, cor_new = subset2evaluate.evaluate.eval_clu_cor(data_new_flat, data_old, metric="human")
        return np.average(clu_new), np.average(cor_new)

    for repetitions, method_kwargs in [
        (100, dict(method="random")),
        (1, dict(method="metric_avg", metric="MetricX-23")),
        (1, dict(method="metric_var", metric="MetricX-23")),
        (1, dict(method="diversity", metric="BLEU")),
        (1, dict(method="metric_cons", metric="MetricX-23")),
        (5, dict(method="pyirt_diffdisc", metric="MetricX-23", model="4pl_score", epochs=1000, retry_on_error=True)),
    ]:
        for _ in range(repetitions):
            data_y = subset2evaluate.select_subset.basic(data_old, **method_kwargs)
            clu_new, cor_new = evaluate_aggregate_second(data_y)
            cor_new_all[method_kwargs["method"]].append(cor_new)
            clu_new_all[method_kwargs["method"]].append(clu_new)


for method in cor_new_all.keys():
    cor_new = cor_new_all[method]
    clu_new = clu_new_all[method]
    print(method, f"COR: {np.average(cor_new):.1%} | CLU: {np.average(clu_new):.2f}")
