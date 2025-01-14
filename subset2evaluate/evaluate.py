from typing import Dict
import numpy as np
import subset2evaluate.utils as utils


def run_evaluate_cluacc(data_new, data_old, metric="human", props=utils.PROPS):
    # both list or descriptor is fine
    data_new = utils.load_data(data_new)
    data_old = utils.load_data(data_old)

    clu_new = []
    acc_new = []
    for prop in props:
        k = int(len(data_old) * prop)
        clu_new.append(eval_subset_clusters(data_new[:k], metric=metric))
        acc_new.append(eval_subset_accuracy(data_new[:k], data_old, metric=metric))

    return clu_new, acc_new


def run_evaluate_top_timebudget(data_old, data_new, metric="human"):
    # both list or descriptor is fine
    data_old = utils.load_data(data_old)
    data_new = utils.load_data(data_new)

    clu_new = []
    acc_new = []
    for prop in utils.PROPS:
        k = int(len(data_old) * prop)
        data_new_inbudget = []
        budget = k
        for item in data_new:
            if item["time"] <= budget:
                budget -= item["time"]
                data_new_inbudget.append(item)
            else:
                break
        clu_new.append(eval_subset_clusters(data_new_inbudget, metric=metric))
        acc_new.append(eval_subset_accuracy(data_new_inbudget, data_old, metric=metric))

    return clu_new, acc_new


def eval_subset_accuracy(data_new: list, data_old: list, metric="human"):
    # evaluates against ordering from data_old
    import itertools
    import numpy as np

    systems = list(data_old[0]["scores"].keys())

    scores_old = get_sys_absolute(data_old, metric=metric)
    scores_new = get_sys_absolute(data_new, metric=metric)

    result = []
    for sys1, sys2 in itertools.combinations(systems, 2):
        result.append((scores_old[sys1] < scores_old[sys2]) == (scores_new[sys1] < scores_new[sys2]))

    return np.average(result)


def eval_subset_clusters(data: list, metric="human"):
    from scipy.stats import wilcoxon
    import warnings
    # computes number of clusters

    # sort from top
    sys_ord = list(get_sys_absolute(data, metric=metric).items())
    sys_ord.sort(key=lambda x: x[1], reverse=True)
    sys_ord = [sys for sys, _ in sys_ord]

    def get_scores(system):
        return [line["scores"][system][metric] for line in data]

    clusters = [[get_scores(sys_ord.pop(0))]]
    while sys_ord:
        sys_scores = get_scores(sys_ord.pop(0))
        diffs = [x - y for x, y in zip(sys_scores, clusters[-1][-1])]
        warnings.simplefilter("ignore", category=UserWarning)
        if all([d == 0 for d in diffs]) or wilcoxon(diffs, alternative="less").pvalue < 0.05:
            clusters.append([sys_scores])
        else:
            clusters[-1].append(sys_scores)
    warnings.resetwarnings()
    return len(clusters)


def get_sys_absolute(data_new, metric="human") -> Dict[str, float]:
    import collections
    import numpy as np

    scores_new = collections.defaultdict(list)

    systems = list(data_new[0]["scores"].keys())
    for line in data_new:
        for sys in systems:
            scores_new[sys].append(line["scores"][sys][metric])

    scores_new = {
        sys: np.average(scores_new[sys])
        for sys in systems
    }

    return scores_new


def get_sys_ordering(data_new: list, metric="human"):
    scores_new = get_sys_absolute(data_new, metric)

    # sort to get ordering
    scores_new = list(scores_new.items())
    # sort from highest
    scores_new.sort(key=lambda x: x[1], reverse=True)

    sys_ord = {
        sys: sys_i
        for sys_i, (sys, sys_v) in enumerate(scores_new)
    }

    return sys_ord


def eval_order_accuracy(scores_new: Dict[str, float], scores_old: Dict[str, float]):
    # evaluates against ordering from data_old
    import itertools
    import numpy as np

    systems = list(scores_old.keys())

    result = []
    for sys1, sys2 in itertools.combinations(systems, 2):
        result.append((scores_old[sys1] < scores_old[sys2]) == (scores_new[sys1] < scores_new[sys2]))

    return np.average(result)


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

    clu_new, acc_new = run_evaluate_cluacc(args.data_old, args.data_new, args.metric)

    print(f"Clusters: {np.average(clu_new):.2f}")
    print(f"Accuracy: {np.average(acc_new):.1%}")
