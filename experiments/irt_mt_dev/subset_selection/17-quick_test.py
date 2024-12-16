# %%
import irt_mt_dev.utils as utils
import numpy as np
import os
import subset2evaluate.evaluate
import subset2evaluate.select_subset

os.chdir("/home/vilda/irt-mt-dev")

data_old = utils.load_data_wmt(year="wmt23", langs="en-cs", normalize=True)
def benchmark_method(repetitions=10, kwargs_dict={}):
    points_y_acc = []
    points_y_clu = []

    # run multiple times to smooth variance
    for _ in range(repetitions):
        clu_new, acc_new = subset2evaluate.evaluate.run_evaluate_topk(
            data_old,
            subset2evaluate.select_subset.run_select_subset(data_old, **kwargs_dict, retry_on_error=False),
            metric="human"
        )
        points_y_acc.append(acc_new)
        points_y_clu.append(clu_new)
        print(f"- ACC: {np.average(acc_new):.2%} | CLU: {np.average(clu_new):.2f}")

    print(f"ACC: {np.average(points_y_acc):.2%} | CLU: {np.average(points_y_clu):.2f}")

# %%
benchmark_method(repetitions=1, kwargs_dict={"method": "pyirt_diff", "model": "amortized_1pl_score", "metric": "MetricX-23-c", "epochs": 100})
benchmark_method(repetitions=1, kwargs_dict={"method": "pyirt_diffdisc", "model": "amortized_4pl_score", "metric": "MetricX-23-c", "epochs": 100})

# %%
print("Random")
benchmark_method(repetitions=10, kwargs_dict={"method": "random"})    

# %%
print("PyIRT-score Fisher Information Content")
benchmark_method(repetitions=10, kwargs_dict={"method": "pyirt_exp", "metric": "MetricX-23-c", "epochs": 1000, "model_type": "4pl_score", "dropout": 0.5, "priors": "hiearchical", "deterministic": True})

# %%
print("NeuralIRT Fisher Information Content")
benchmark_method(repetitions=3, kwargs_dict={"method": "nnirt_fic", "metric": "MetricX-23-c", "max_epochs": 1000})

# %%
print("COMET-var")
benchmark_method(repetitions=1, kwargs_dict={"method": "comet_var"})

# %%
print("COMET-avg")
benchmark_method(repetitions=1, kwargs_dict={"method": "comet_avg"})

# %%
print("Human-var")
benchmark_method(repetitions=1, kwargs_dict={"method": "var", "metric": "human"})
print("Human-avg")
benchmark_method(repetitions=1, kwargs_dict={"method": "avg", "metric": "human"})