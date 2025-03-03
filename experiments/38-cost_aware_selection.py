# %%

import tqdm
import subset2evaluate.utils as utils
import numpy as np
import subset2evaluate.evaluate
import subset2evaluate.select_subset
import subset2evaluate.utils

data_old_all = list(utils.load_data_wmt_all(normalize=True).values())[:9]

for repetitions, method_kwargs in [
    (10, dict(method="random")),
    (1, dict(method="metric_avg", metric="MetricX-23")),
    (1, dict(method="metric_var", metric="MetricX-23")),
    (1, dict(method="metric_cons", metric="MetricX-23")),
    (1, dict(method="diversity", metric="BLEU")),
    (5, dict(method="pyirt_diffdisc", metric="MetricX-23", retry_on_error=True)),
]:
    clu_all = []
    cor_all = []
    for data_old in tqdm.tqdm(data_old_all):
        for repetition in range(repetitions):
            data_new_raw = subset2evaluate.select_subset.basic(data_old, **method_kwargs)

            for prop in subset2evaluate.utils.PROPS:
                B = int(prop * len(data_old))

                data_new = subset2evaluate.select_subset.costaware(data_new_raw, B)
                clu_new = subset2evaluate.evaluate.eval_subset_clusters(data_new)
                cor_new = subset2evaluate.evaluate.eval_subset_correlation(data_new, data_old)

                clu_all.append(clu_new)
                cor_all.append(cor_new)

    print(f"{method_kwargs['method']:>15}", f"{np.average(clu_all):.2f} {np.average(cor_all):.1%}")
