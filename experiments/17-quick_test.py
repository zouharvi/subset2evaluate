# %%
import tqdm
import subset2evaluate.utils as utils
import numpy as np
import subset2evaluate.evaluate
import subset2evaluate.select_subset
# %%

def benchmark_method(repetitions=10, kwargs_dict={}):
    data_old = utils.load_data_wmt(year="wmt23", langs="en-cs", normalize=True)
    points_y_acc = []
    points_y_clu = []

    # run multiple times to smooth variance
    for _ in range(repetitions):
        clu_new, cor_new = subset2evaluate.evaluate.eval_clucor(
            subset2evaluate.select_subset.basic(data_old, **kwargs_dict, retry_on_error=False),
            data_old,
            metric="human"
        )
        points_y_acc.append(cor_new)
        points_y_clu.append(clu_new)
        print(f"- ACC: {np.average(cor_new):.1%} | CLU: {np.average(clu_new):.2f}")

    if repetitions > 1:
        print(f"ACC: {np.average(points_y_acc):.1%} | CLU: {np.average(points_y_clu):.2f}")


data_old_all = list(utils.load_data_wmt_all(normalize=True).values())[:9]


def benchmark_method_all(repetitions=10, kwargs_dict={}):
    points_y_acc = []
    points_y_clu = []

    for data_old in tqdm.tqdm(data_old_all):
        # run multiple times to smooth variance
        for _ in range(repetitions):
            clu_new, cor_new = subset2evaluate.evaluate.eval_clucor(
                subset2evaluate.select_subset.basic(data_old, **kwargs_dict, retry_on_error=False),
                data_old,
                metric="human"
            )
            points_y_acc.append(cor_new)
            points_y_clu.append(clu_new)

    print(f"ACC: {np.average(points_y_acc):.1%} | CLU: {np.average(points_y_clu):.2f}")


# %%
print("PyIRT-score Fisher Information Content")
benchmark_method(repetitions=10, kwargs_dict={"method": "pyirt_fic", "metric": "MetricX-23-c", "epochs": 1000, "model_type": "4pl_score"})

# %%
# works a bit worse
print("PyIRT-score Fisher Information Content with enforced positive discrimination")
benchmark_method(repetitions=10, kwargs_dict={"enforce_positive_disc": True, "method": "pyirt_fic", "metric": "MetricX-23-c", "epochs": 1000, "model_type": "4pl_score"})

# %%
print("Random")
# benchmark_method(repetitions=10, kwargs_dict={"method": "random"})
benchmark_method_all(repetitions=10, kwargs_dict={"method": "random"})

# %%
print("Pointwise alignment")
benchmark_method_all(repetitions=1, kwargs_dict={"method": "pointwise_alignment", "metric": "MetricX-23"})
benchmark_method_all(repetitions=1, kwargs_dict={"method": "pointwise_alignment", "metric": "human"})

# %%
print("PreCOMET-{avg,var}")
benchmark_method(repetitions=1, kwargs_dict={"method": "comet_var"})
benchmark_method(repetitions=1, kwargs_dict={"method": "comet_avg"})

# %%
print("Human-var")
benchmark_method(repetitions=1, kwargs_dict={"method": "metric_var", "metric": "human"})
print("Human-avg")
benchmark_method(repetitions=1, kwargs_dict={"method": "metric_avg", "metric": "human"})

# %%
print("MetricX-var")
benchmark_method_all(repetitions=1, kwargs_dict={"method": "metric_var", "metric": "MetricX-23"})
print("MetricX-avg")
benchmark_method_all(repetitions=1, kwargs_dict={"method": "metric_avg", "metric": "MetricX-23"})

# %%
print("PreCOMET variants")
benchmark_method(repetitions=1, kwargs_dict={"method": "precomet_var"})
benchmark_method(repetitions=1, kwargs_dict={"method": "precomet_avg"})
benchmark_method(repetitions=1, kwargs_dict={"method": "precomet_diff"})
benchmark_method(repetitions=1, kwargs_dict={"method": "precomet_disc"})
benchmark_method(repetitions=1, kwargs_dict={"method": "precomet_diffdisc"})
