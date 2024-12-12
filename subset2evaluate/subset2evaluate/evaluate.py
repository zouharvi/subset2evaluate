import numpy as np
import subset2evaluate.utils as utils
import irt_mt_dev.utils

def run_evaluate_topk(data_old, data_new, metric="human"):
    # both list or descriptor is fine
    data_old = utils.load_data(data_old)
    data_new = utils.load_data(data_new)

    clu_old = []
    clu_new = []
    acc_new = []
    clu_old.append(irt_mt_dev.utils.eval_system_clusters(data_old, metric=metric))
    for prop in irt_mt_dev.utils.PROPS:
        k = int(len(data_old)*prop)
        clu_new.append(irt_mt_dev.utils.eval_system_clusters(data_new[:k], metric=metric))
        acc_new.append(irt_mt_dev.utils.eval_subset_accuracy(data_new[:k], data_old, metric=metric))

    return (clu_old, clu_new), acc_new

def run_evaluate_top_timebudget(data_old, data_new, metric="human"):
    # both list or descriptor is fine
    data_old = utils.load_data(data_old)
    data_new = utils.load_data(data_new)

    clu_old = []
    clu_new = []
    acc_new = []
    clu_old.append(irt_mt_dev.utils.eval_system_clusters(data_old, metric=metric))
    for prop in irt_mt_dev.utils.PROPS:
        k = int(len(data_old)*prop)
        data_new_inbudget = []
        budget = k
        for item in data_new:
            if item["time"] <= budget:
                budget -= item["time"]
                data_new_inbudget.append(item)
            else:
                break
        clu_new.append(irt_mt_dev.utils.eval_system_clusters(data_new_inbudget, metric=metric))
        acc_new.append(irt_mt_dev.utils.eval_subset_accuracy(data_new_inbudget, data_old, metric=metric))

    return (clu_old, clu_new), acc_new

if __name__ == "__main__":
    import argparse

    args = argparse.ArgumentParser()
    args.add_argument('data_old', type=str, default='wmt23/en-cs')
    args.add_argument('data_new', type=str, default='wmt23/en-cs')
    args.add_argument('--metric', type=str, default='human')
    args = args.parse_args()

    (clu_old, clu_new), acc_new = run_evaluate_topk(args.data_old, args.data_new, args.metric)

    print(f"Clusters (old->new): {np.average(clu_old):.3f} -> {np.average(clu_new):.3f}")
    print(f"Accuracy (new): {np.average(acc_new):.2%}")