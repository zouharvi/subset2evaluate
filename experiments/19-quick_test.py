# %%
import tqdm
import subset2evaluate.utils as utils
import numpy as np
import subset2evaluate.evaluate
import subset2evaluate.select_subset


def benchmark_method(repetitions=10, kwargs_dict={}):
    data_old = utils.load_data_wmt(year="wmt23", langs="en-cs", normalize=True)
    points_y_cor = []
    points_y_clu = []

    # run multiple times to smooth variance
    for _ in range(repetitions):
        clu_new, cor_new = subset2evaluate.evaluate.eval_clu_cor(
            subset2evaluate.select_subset.basic(data_old, **kwargs_dict, retry_on_error=False),
            data_old,
            metric="human"
        )
        points_y_cor.append(cor_new)
        points_y_clu.append(clu_new)
        print(f"- COR: {np.average(cor_new):.1%} | CLU: {np.average(clu_new):.2f}")

    if repetitions > 1:
        print(f"COR: {np.average(points_y_cor):.1%} | CLU: {np.average(points_y_clu):.2f}")


data_old_all = list(utils.load_data_wmt_all(normalize=True).values())[:9]


def benchmark_method_all(repetitions=10, kwargs_dict={}):
    points_y_cor = []
    points_y_clu = []

    for data_old in tqdm.tqdm(data_old_all):
        # run multiple times to smooth variance
        for _ in range(repetitions):
            data_new = subset2evaluate.select_subset.basic(data_old, **kwargs_dict, retry_on_error=False)
            clu_new, cor_new = subset2evaluate.evaluate.eval_clu_cor(
                data_new,
                data_old,
                metric="human"
            )
            points_y_cor.append(cor_new)
            points_y_clu.append(clu_new)

    print(kwargs_dict["method"], f"COR: {np.average(points_y_cor):.1%} | CLU: {np.average(points_y_clu):.2f}")


# %%
benchmark_method_all(repetitions=10, kwargs_dict={"method": "random"})

# %%
benchmark_method_all(repetitions=1, kwargs_dict={"method": "metric_cons", "metric": "MetricX-23"})
benchmark_method_all(repetitions=1, kwargs_dict={"method": "metric_var", "metric": "MetricX-23"})
benchmark_method_all(repetitions=1, kwargs_dict={"method": "metric_avg", "metric": "MetricX-23"})
