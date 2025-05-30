# %%

import tqdm
import subset2evaluate.utils as utils
import numpy as np
import subset2evaluate.evaluate
import subset2evaluate.select_subset
import subset2evaluate.utils

data_old_all = list(utils.load_data_wmt_test(normalize=True).values())

spa_all_random = []

for repetitions, method_kwargs in [
    (100, dict(method="random")),
    (1, dict(method="metric_avg", metric="MetricX-23")),
    (1, dict(method="metric_var", metric="MetricX-23")),
    (1, dict(method="metric_cons", metric="MetricX-23")),
    (1, dict(method="diversity", metric="LM")),
    (5, dict(method="pyirt_diffdisc", metric="MetricX-23-c", retry_on_error=True)),
    (1, dict(method="precomet_avg")),
    (1, dict(method="precomet_var")),
    (1, dict(method="precomet_cons")),
    (1, dict(method="precomet_diversity")),
    (1, dict(method="precomet_diffdisc_direct")),
]:
    spa_all = []
    for _ in tqdm.tqdm(range(repetitions)):
        for data_old in data_old_all:
            data_new_raw = subset2evaluate.select_subset.basic(data_old, **method_kwargs)

            for prop in subset2evaluate.utils.PROPS:
                B = int(prop * len(data_old))

                data_new = subset2evaluate.select_subset.costaware(data_new_raw, B)
                spa_new = subset2evaluate.evaluate.eval_subset_spa(data_new, data_old)

                spa_all.append(spa_new)
                if method_kwargs["method"] == "random":
                    spa_all_random.append(spa_new)

    print(f"{method_kwargs['method']:>15}", f"{np.average(spa_all):.1%}")


# %%

import subset2evaluate.utils

spa_all_random_arr = np.array(spa_all_random).reshape(100, -1, len(subset2evaluate.utils.PROPS)).mean(axis=(1, 2))
conf = subset2evaluate.utils.confidence_interval(spa_all_random_arr, confidence=0.90)
print(f"{(conf[1]-conf[0])/2:.2%}")