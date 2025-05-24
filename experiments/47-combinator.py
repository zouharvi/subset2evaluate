# %%
import tqdm
import subset2evaluate.utils as utils
import numpy as np
import subset2evaluate.evaluate
import subset2evaluate.select_subset

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

benchmark_method_all(
    repetitions=1, kwargs_dict=dict(method="diversity", metric="BLEU")
)

# %%
benchmark_method_all(
    repetitions=1, kwargs_dict=dict(method="metric_var", metric="MetricX-23")
)
benchmark_method_all(
    repetitions=1, kwargs_dict=dict(method="metric_var", metric="XCOMET-XL")
)
benchmark_method_all(
    repetitions=1, kwargs_dict={
        "method": "combinator",
        "operation": "mul",
        "normalize_zscore": True,
        "normalize_01": True,
        "methods": [
            dict(method="metric_var", metric="MetricX-23"),
            dict(method="metric_var", metric="XCOMET-XL"),
        ]
    }
)


# %%
benchmark_method_all(
    repetitions=1, kwargs_dict=dict(method="metric_cons", metric="MetricX-23")
)
benchmark_method_all(
    repetitions=1, kwargs_dict=dict(method="metric_cons", metric="XCOMET-XL")
)
benchmark_method_all(
    repetitions=1, kwargs_dict={
        "method": "combinator",
        "operation": "mul",
        "normalize_zscore": True,
        "normalize_01": True,
        "methods": [
            dict(method="metric_cons", metric="MetricX-23"),
            dict(method="metric_cons", metric="XCOMET-XL"),
        ]
    }
)
# %%

benchmark_method_all(
    repetitions=1, kwargs_dict={
        "method": "combinator",
        "operation": "mul",
        "normalize_zscore": True,
        "normalize_01": True,
        "methods": [
            dict(method="metric_var", metric="MetricX-23"),
            dict(method="metric_var", metric="XCOMET-XL"),
            dict(method="metric_cons", metric="MetricX-23"),
            dict(method="metric_var", metric="XCOMET-XL"),
        ]
    }
)

# %%

benchmark_method_all(
    repetitions=1, kwargs_dict={
        "method": "combinator",
        "operation": "mul",
        "normalize_zscore": True,
        "normalize_01": True,
        "methods": [
            dict(method="metric_var", metric="MetricX-23"),
            dict(method="metric_var", metric="XCOMET-XL"),
            dict(method="metric_cons", metric="MetricX-23"),
            dict(method="metric_var", metric="XCOMET-XL"),
            dict(method="diversity", metric="BLEU"),
        ]
    }
)

# %%

subset2evaluate.evaluate.eval_metrics_correlations(data_old_all[0], metric_target="human")