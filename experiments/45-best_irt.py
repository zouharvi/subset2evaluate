# %%

import tqdm
import subset2evaluate.utils as utils
import numpy as np
import subset2evaluate.evaluate
import subset2evaluate.select_subset

data_old_all = list(utils.load_data_wmt_all(normalize=True).values())[:9]


def benchmark_method_all(repetitions=1, kwargs_dict={}):
    points_y_cor = []
    points_y_clu = []

    for data_old in data_old_all:
        # run multiple times to smooth variance
        for _ in range(repetitions):
            data_new = subset2evaluate.select_subset.basic(data_old, **kwargs_dict)
            clu_new, cor_new = subset2evaluate.evaluate.eval_clucor(
                data_new,
                data_old,
                metric="human"
            )
            points_y_cor.append(cor_new)
            points_y_clu.append(clu_new)

    return np.average(points_y_cor), np.average(points_y_clu)


results = []
for epochs in tqdm.tqdm(range(500, 2000, 200)):
    out = benchmark_method_all(repetitions=1, kwargs_dict={"method": "pyirt_diffdisc", "model": "4pl_score", "metric": "MetricX-23", "epochs": epochs, "retry_on_error": True})
    results.append((epochs, out))
    print(f"Epochs: {epochs} | COR: {out[0]:.1%} | CLU: {out[1]:.2f}")
