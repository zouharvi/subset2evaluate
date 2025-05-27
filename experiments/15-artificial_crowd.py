# %%
import subset2evaluate.select_subset
import subset2evaluate.evaluate
import subset2evaluate.utils
import random
import tqdm
import numpy as np
import subset2evaluate.utils as utils


data_old_all = list(utils.load_data_wmt_test(normalize=True).items())

# number of models in each language pair
print([len(data_old[0]["scores"]) for data_old_name, data_old in data_old_all])
SUBSET_SIZE = 4

# %%

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
    for _ in tqdm.tqdm(range(repetitions), desc="Repetition"):
        R_SAMPLE = random.Random(0)
        for data_old_name, data_old in data_old_all:
            models = list(data_old[0]["scores"].keys())
            models_artificial = R_SAMPLE.sample(list(sorted(models)), k=SUBSET_SIZE)
            models_true = sorted(set(models) - set(models_artificial))

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
            data_new, load_model = subset2evaluate.select_subset.basic(
                data_old_artificial,
                **method_kwargs,
                return_model=True,
                load_model=load_model if method_kwargs["method"] != "pyirt_diffdisc" else None,
            )
            spa_all.append(subset2evaluate.evaluate.eval_spa(
                [
                    data_old_i_to_line[line["i"]]
                    for line in data_new
                ],
                data_old_true,
                metric="human"
            ))
    print(method_kwargs["method"], f"{np.average(spa_all):.1%}")
