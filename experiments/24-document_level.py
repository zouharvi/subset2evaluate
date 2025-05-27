# %%
import collections
import subset2evaluate.utils as utils
import subset2evaluate.evaluate
import subset2evaluate.select_subset
import numpy as np
import tqdm

data_old_all = list(utils.load_data_wmt_test(normalize=True).values())

# %%
spa_new_all = collections.defaultdict(list)

for repetitions, method_kwargs in [
    (100, dict(method="random")),
    (1, dict(method="metric_avg", metric="MetricX-23-c")),
    (1, dict(method="metric_var", metric="MetricX-23-c")),
    (1, dict(method="metric_cons", metric="MetricX-23-c")),
    (1, dict(method="diversity", metric="lm")),
    (5, dict(method="pyirt_diffdisc", metric="MetricX-23-c", model="4pl_score", retry_on_error=True)),
    (1, dict(method="precomet_avg")),
    (1, dict(method="precomet_var")),
    (1, dict(method="precomet_cons")),
    (1, dict(method="precomet_diversity")),
    (1, dict(method="precomet_diffdisc_direct")),
]:
    load_model = None
    spa_all = []
    for data_old in tqdm.tqdm(data_old_all):
        for _ in range(repetitions):
            def evaluate_aggregate_doc(data_scored):
                data_old_aggregated = collections.defaultdict(list)
                for line in data_scored:
                    data_old_aggregated[line["doc"]].append(line)

                data_old_aggregated = [
                    {
                        "doc": doc,
                        "i": [line["i"] for line in lines],
                        # average the utilities across the document
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
                return subset2evaluate.evaluate.eval_spa(data_new_flat, data_old, metric="human")

            data_y, load_model = subset2evaluate.select_subset.basic(
                data_old,
                **method_kwargs,
                load_model=load_model if method_kwargs["method"] != "pyirt_diffdisc" else None,
                return_model=True
            )
            spa_all.append(evaluate_aggregate_doc(data_y))

    print(method_kwargs["method"], f"{np.average(spa_all):.1%}")