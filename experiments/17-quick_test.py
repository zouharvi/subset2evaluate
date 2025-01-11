# %%
import tqdm
import subset2evaluate.utils as utils
import numpy as np
import subset2evaluate.evaluate
import subset2evaluate.select_subset

def benchmark_method(repetitions=10, kwargs_dict={}):
    data_old = utils.load_data_wmt(year="wmt23", langs="en-cs", normalize=True)
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
        print(f"- ACC: {np.average(acc_new):.1%} | CLU: {np.average(clu_new):.2f}")

    if repetitions > 1:
        print(f"ACC: {np.average(points_y_acc):.1%} | CLU: {np.average(points_y_clu):.2f}")

data_old_all = list(utils.load_data_wmt_all(normalize=True).values())[:9]
def benchmark_method_all(repetitions=10, kwargs_dict={}):
    points_y_acc = []
    points_y_clu = []

    for data_old in tqdm.tqdm(data_old_all):
        # run multiple times to smooth variance
        for _ in range(repetitions):
            clu_new, acc_new = subset2evaluate.evaluate.run_evaluate_topk(
                data_old,
                subset2evaluate.select_subset.run_select_subset(data_old, **kwargs_dict, retry_on_error=False),
                metric="human"
            )
            points_y_acc.append(acc_new)
            points_y_clu.append(clu_new)

    print(f"ACC: {np.average(points_y_acc):.1%} | CLU: {np.average(points_y_clu):.2f}")

# with step size 20 and 10 samplings
# - ACC: 87.71% | CLU: 1.10


# %%
print("Experimental")

# ACC: 81.25% | CLU: 1.88
benchmark_method_all(repetitions=1, kwargs_dict={"method": "precomet_diff_precomet_disc", "reverse": False})
# ACC: 93.09% | CLU: 3.22
benchmark_method_all(repetitions=1, kwargs_dict={"method": "precomet_diff_precomet_disc", "reverse": True})
# ACC: 92.81% | CLU: 3.01
benchmark_method_all(repetitions=1, kwargs_dict={"method": "precomet_diffdisc", "reverse": True})

# %%
print("PyIRT-score Fisher Information Content")
benchmark_method(repetitions=10, kwargs_dict={"method": "pyirt_fic", "metric": "MetricX-23-c", "epochs": 1000, "model_type": "4pl_score"})

# %%
# works a bit worse
print("PyIRT-score Fisher Information Content with enforced positive discrimination")
benchmark_method(repetitions=10, kwargs_dict={"enforce_positive_disc": True, "method": "pyirt_fic", "metric": "MetricX-23-c", "epochs": 1000, "model_type": "4pl_score"})

# %%
print("Random")
benchmark_method(repetitions=10, kwargs_dict={"method": "random"})    

# %%
print("PreCOMET-{avg,var}")
benchmark_method(repetitions=1, kwargs_dict={"method": "comet_var"})
benchmark_method(repetitions=1, kwargs_dict={"method": "comet_avg"})

# %%
print("Human-var")
benchmark_method(repetitions=1, kwargs_dict={"method": "var", "metric": "human"})
print("Human-avg")
benchmark_method(repetitions=1, kwargs_dict={"method": "avg", "metric": "human"})

# %%
print("PreCOMET variants")
benchmark_method(repetitions=1, kwargs_dict={"method": "precomet_var"})
benchmark_method(repetitions=1, kwargs_dict={"method": "precomet_avg"})
benchmark_method(repetitions=1, kwargs_dict={"method": "precomet_diff"})
benchmark_method(repetitions=1, kwargs_dict={"method": "precomet_disc"})
benchmark_method(repetitions=1, kwargs_dict={"method": "precomet_diffdisc"})