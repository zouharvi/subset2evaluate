# %%
import collections
import subset2evaluate.select_subset
import subset2evaluate.evaluate
import subset2evaluate.utils
import random
import tqdm
import numpy as np
import subset2evaluate.utils as utils

random.seed(0)

data_old_all = list(utils.load_data_wmt_all(normalize=True).items())[:9]

# number of models in each language pair
print([len(data_old[0]["scores"]) for data_old_name, data_old in data_old_all])
SUBSET_SIZE = [4]

# %%
cors_all = collections.defaultdict(lambda: collections.defaultdict(list))
clus_all = collections.defaultdict(lambda: collections.defaultdict(list))
cache_model = collections.defaultdict(lambda: None)

# %%

for subset_size in tqdm.tqdm(SUBSET_SIZE, desc="Subset size"):
    # run multiple times
    for data_old_name, data_old in tqdm.tqdm(data_old_all, desc="Language pair"):
        for _ in tqdm.tqdm(range(5), desc="Repetition"):
            models = list(data_old[0]["scores"].keys())
            models_true = random.sample(models, k=8)
            models_artificial = random.sample(sorted(set(models) - set(models_true)), k=subset_size)

            data_old_true = [
                {
                    **line,
                    "scores": {
                        model: v
                        for model, v in line["scores"].items()
                        if model in models_true
                    },
                    "tgt": {
                        model: v
                        for model, v in line["tgt"].items()
                        if model in models_true
                    }
                }
                for line in data_old
            ]
            data_old_i_to_line = {line["i"]: line for line in data_old_true}
            data_old_artificial = [
                {
                    **line,
                    "scores": {
                        model: v
                        for model, v in line["scores"].items()
                        if model in models_artificial
                    },
                    "tgt": {
                        model: v
                        for model, v in line["tgt"].items()
                        if model in models_artificial
                    }
                }
                for line in data_old
            ]

            # we dropped some models but we can recover them with the same ordering from data_old
            for cache, method_kwargs in [
                (False, dict(method="random")),
                (True, dict(method="cometsrc_var")),
                (True, dict(method="cometsrc_avg")),
                (True, dict(method="cometsrc_diversity")),
                (True, dict(method="local_cometsrc_diffdisc")),
                (True, dict(method="local_cometsrc_cons")),
                (False, dict(method="metric_var", metric="MetricX-23-c")),
                (False, dict(method="metric_avg", metric="MetricX-23-c")),
                (False, dict(method="metric_cons", metric="MetricX-23-c")),
                (False, dict(method="diversity", method="BLEU")),
                (False, dict(method="pyirt_diffdisc", model="4pl_score", metric="MetricX-23-c")),
            ]:
                data_new, load_model = subset2evaluate.select_subset.basic(
                    data_old_artificial, **method_kwargs,
                    retry_on_error=True, return_model=True,
                    load_model=cache_model[method_kwargs["method"]],
                )

                clu_new, cor_new = subset2evaluate.evaluate.eval_clucor(
                    [
                        data_old_i_to_line[line["i"]]
                        for line in data_new
                    ],
                    data_old_true,
                    metric="human"
                )
                clus_all[method_kwargs["method"]][subset_size].append(clu_new)
                cors_all[method_kwargs["method"]][subset_size].append(cor_new)


# %%
def method_formatter(method):
    DICT = {
        "random": "Random",
        "cometsrc_var": r"MetricVar\textsuperscript{src}",
        "cometsrc_avg": r"MetricAvg\textsuperscript{src}",
        "cometsrc_diversity": r"Diversity\textsuperscript{src}",
        "metric_var": "MetricVar",
        "metric_avg": "MetricAvg",
        "metric_cons": "MetricCons",
        "diversity": "Diversity",
        "pyirt_diffdisc": r"Diff.$\times$Disc.",
        "local_cometsrc_diffdisc": r"Diff.\textsuperscript{src}$\times$Disc\textsuperscript{src}",
        "local_cometsrc_cons": r"MetricCons\textsuperscript{src}",
    }
    if method in DICT:
        return DICT[method]
    else:
        return method


# print results
for subset_size in SUBSET_SIZE:
    for method in cors_all.keys():
        print(method_formatter(method), end=" & ")
        print(
            f"{np.average(cors_all[method][subset_size]):.1%}".replace("%", r"\%"),
            f"{np.average(clus_all[method][subset_size]):.2f}".replace("%", r"\%"),
            sep=" & ",
            end=" \\\\\n"
        )
