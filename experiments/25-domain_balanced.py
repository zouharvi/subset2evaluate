# %%
import collections
import subset2evaluate.utils as utils
import subset2evaluate.evaluate
import subset2evaluate.select_subset
import numpy as np
import tqdm
import itertools

data_old_all = list(utils.load_data_wmt_test(normalize=True).values())

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
    spa_all = []
    load_model = None
    for data_old in tqdm.tqdm(data_old_all):
        for _ in range(repetitions):
            def evaluate_balanced_domains(data_scored):
                data_aggregated = collections.defaultdict(list)
                for line in data_scored:
                    data_aggregated[line["domain"]].append(line)

                # sort within each domain by utility
                data_aggregated = [
                    sorted(domain, key=lambda x: x["subset2evaluate_utility"], reverse=True)
                    for domain in data_aggregated.values()
                ]

                # interveawe the lists
                # this is a cheap trick that somewhat guarantees that data_new_flat[:k] has balanced domains
                # create a list of ABCDABCDABCDCDCDDDD
                data_new_flat = [doc for docs in itertools.zip_longest(*data_aggregated) for doc in docs]
                data_new_flat = [line for line in data_new_flat if line is not None]
                return subset2evaluate.evaluate.eval_spa(data_new_flat, data_old, metric="human")

            data_y, load_model = subset2evaluate.select_subset.basic(
                data_old,
                **method_kwargs,
                load_model=load_model if method_kwargs["method"] != "pyirt_diffdisc" else None,
                return_model=True
            )
            spa_all.append(evaluate_balanced_domains(data_y))
    print(method_kwargs["method"], f"{np.average(spa_all):.1%}")