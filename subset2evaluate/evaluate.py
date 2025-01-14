from typing import Dict, List, Tuple
import numpy as np
import subset2evaluate
import subset2evaluate.utils as utils


def run_evaluate_cluacc(data_new: List[Dict], data_old: List[Dict], metric="human", props: List[float]=utils.PROPS) -> Tuple[float, float]:
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


def run_evaluate_cluacc_par(
        data_new: List[Dict],
        data_old: List[Dict],
        clus_tgt: List[float],
        accs_tgt: List[float],
        metric="human",
        props: List[float]=utils.PROPS,
        workers=10,
) -> Tuple[float, float]:
    """
    Evaluates the proportion of data that is needed to achieve parity with target.
    """
    import multiprocessing.pool

    # both list or descriptor is fine
    data_new = utils.load_data(data_new)
    data_old = utils.load_data(data_old)

    def _par_clu(data_new, clu_tgt, metric):
        for k in range(5, len(data_new) + 1):
            if eval_subset_clusters(data_new[:k], metric=metric) >= clu_tgt:
                break
        return k
    
    def _par_acc(data_new, data_old, acc_tgt, metric):
        for k in range(5, len(data_new) + 1):
            if eval_subset_accuracy(data_new[:k], data_old, metric=metric) >= acc_tgt:
                break
        return k

    # multiprocess for each prop rather than k because the thread
    # orchestration would be more expensive otherwise
    with multiprocessing.pool.ThreadPool(min(workers, len(props))) as pool:
        ks_clu_par = pool.starmap(
            _par_clu,
            [(data_new, clu_tgt, metric) for prop, clu_tgt in zip(props, clus_tgt)]
        )
        ks_clu_par = [k / (len(data_old) * prop) for k, prop in zip(ks_clu_par, props)]

        ks_acc_par = pool.starmap(
            _par_acc,
            [(data_new, data_old, clu_tgt, metric) for prop, clu_tgt in zip(props, accs_tgt)]
        )
        ks_acc_par = [k / (len(data_old) * prop) for k, prop in zip(ks_acc_par, props)]
    
    return np.average(ks_clu_par), np.average(ks_acc_par)


def precompute_randnorm(
    data_old: List[Dict],
    random_seeds=10,
    metric="human",
    workers=10,
) -> Tuple[List[float], List[float], float, float]:
    import subset2evaluate.select_subset

    clu_random = []
    acc_random = []
    for seed in range(random_seeds):
        clu_new, acc_new = run_evaluate_cluacc(
            subset2evaluate.select_subset.run_select_subset(data_old, method="random", seed=seed),
            data_old,
            metric=metric,
        )
        clu_random.append(clu_new)
        acc_random.append(acc_new)
    clu_random = np.average(clu_random, axis=0)
    acc_random = np.average(acc_random, axis=0)

    pars_clu_rand = []
    pars_acc_rand = []

    for seed in range(random_seeds, 2*random_seeds):
        par_clu_rand, par_acc_rand = run_evaluate_cluacc_par(
            subset2evaluate.select_subset.run_select_subset(data_old, method="random", seed=seed),
            data_old,
            clu_random,
            acc_random,
            metric=metric,
            workers=workers,
        )
        pars_clu_rand.append(par_clu_rand)
        pars_acc_rand.append(par_acc_rand)

    return (clu_random, acc_random), (np.average(pars_clu_rand), np.average(pars_acc_rand))

def run_evaluate_cluacc_randnorm(
    data_new: List[Dict],
    data_old: List[Dict],
    random_seeds=10,
    metric="human",
    cluacc_precomputed = None
) -> Tuple[float, float]:

    if cluacc_precomputed is not None:
        (clu_random, acc_random), (clu_random_norm, acc_random_norm) = cluacc_precomputed
    else:
        (clu_random, acc_random), (clu_random_norm, acc_random_norm) = precompute_randnorm(data_old, random_seeds=random_seeds, metric=metric)

    # compute the parity of the new data
    par_clu, par_acc = run_evaluate_cluacc_par(
        data_new, data_old,
        clu_random, acc_random,
        metric=metric
    )

    return par_clu/clu_random_norm, par_acc/acc_random_norm


def run_evaluate_top_timebudget(data_new, data_old, metric="human"):
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


def eval_subset_accuracy(data_new: List[Dict], data_old: List[Dict], metric="human"):
    # evaluates against ordering from data_old
    import itertools

    systems = list(data_old[0]["scores"].keys())

    scores_old = get_sys_absolute(data_old, metric=metric)
    scores_new = get_sys_absolute(data_new, metric=metric)

    result = []
    for sys1, sys2 in itertools.combinations(systems, 2):
        result.append((scores_old[sys1] < scores_old[sys2]) == (scores_new[sys1] < scores_new[sys2]))

    return np.average(result)


def eval_subset_clusters(data: List[Dict], metric="human"):
    from scipy.stats import wilcoxon
    import warnings

    # if we have just 3 samples, we can't say that there are clusters
    if len(data) < 3:
        return 1

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

        with warnings.catch_warnings(action="ignore"):
            if all([d == 0 for d in diffs]) or wilcoxon(diffs, alternative="less").pvalue < 0.05:
                clusters.append([sys_scores])
            else:
                clusters[-1].append(sys_scores)
                
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


def get_sys_ordering(data_new: List[Dict], metric="human"):
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
