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

    load_model = None

    for data_old in tqdm.tqdm(data_old_all):
        # run multiple times to smooth variance
        for _ in range(repetitions):
            data_new, load_model = subset2evaluate.select_subset.basic(data_old, **kwargs_dict, retry_on_error=False, load_model=load_model, return_model=True)
            clu_new, cor_new = subset2evaluate.evaluate.eval_clucor(
                data_new,
                data_old,
                metric="human"
            )
            points_y_acc.append(cor_new)
            points_y_clu.append(clu_new)

    print(f"ACC: {np.average(points_y_acc):.1%} | CLU: {np.average(points_y_clu):.2f}")


# %%
print("Random")
# benchmark_method(repetitions=10, kwargs_dict={"method": "random"})
benchmark_method_all(repetitions=10, kwargs_dict={"method": "random"})

# %%
print("K-means")
benchmark_method_all(repetitions=1, kwargs_dict={"method": "kmeans"})