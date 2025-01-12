import numpy as np
import subset2evaluate.utils as utils

def run_evaluate_topk(data_old, data_new, metric="human"):
    # both list or descriptor is fine
    data_old = utils.load_data(data_old)
    data_new = utils.load_data(data_new)

    clu_new = []
    acc_new = []
    for prop in utils.PROPS:
        k = int(len(data_old)*prop)
        clu_new.append(utils.eval_system_clusters(data_new[:k], metric=metric))
        acc_new.append(utils.eval_subset_accuracy(data_new[:k], data_old, metric=metric))

    return clu_new, acc_new

def run_evaluate_top_timebudget(data_old, data_new, metric="human"):
    # both list or descriptor is fine
    data_old = utils.load_data(data_old)
    data_new = utils.load_data(data_new)

    clu_new = []
    acc_new = []
    for prop in utils.PROPS:
        k = int(len(data_old)*prop)
        data_new_inbudget = []
        budget = k
        for item in data_new:
            if item["time"] <= budget:
                budget -= item["time"]
                data_new_inbudget.append(item)
            else:
                break
        clu_new.append(utils.eval_system_clusters(data_new_inbudget, metric=metric))
        acc_new.append(utils.eval_subset_accuracy(data_new_inbudget, data_old, metric=metric))

    return clu_new, acc_new

def main_cli():
    import argparse

    args = argparse.ArgumentParser(
        description="Meta-evaluate subset selection methods with cluster count and system accuracy."
    )
    args.add_argument(
        'data_old', type=str, default='wmt23/en-cs',
        help="Original data descriptor or path."
    )
    args.add_argument(
        'data_new', type=str, default='wmt23/en-cs',
        help="Path to new ordered data."
    )
    args.add_argument(
        '--metric', type=str, default='human',
        help="Metric to evaluate against, e.g., human or human_consistency. Can also be a metric and not human score."
    )
    args = args.parse_args()

    clu_new, acc_new = run_evaluate_topk(args.data_old, args.data_new, args.metric)

    print(f"Clusters: {np.average(clu_new):.2f}")
    print(f"Accuracy: {np.average(acc_new):.1%}")