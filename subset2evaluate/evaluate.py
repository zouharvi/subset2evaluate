from typing import Dict, List, Tuple
import numpy as np
import subset2evaluate.utils as utils


def eval_clucor(
        data_new: List[Dict],
        data_old: List[Dict],
        metric="human",
        props: List[float] = utils.PROPS
) -> Tuple[float, float]:
    # both list or descriptor is fine
    data_new = utils.load_data(data_new)
    data_old = utils.load_data(data_old)

    clu_new = []
    cor_new = []
    for prop in props:
        k = int(len(data_old) * prop)
        clu_new.append(eval_subset_clusters(data_new[:k], metric=metric))
        cor_new.append(eval_subset_correlation(data_new[:k], data_old, metric=metric))

    return clu_new, cor_new


def eval_clucor_par(
    data_new: List[Dict],
    data_old: List[Dict],
    clus_tgt: List[float],
    cors_tgt: List[float],
    metric="human",
    props: List[float] = utils.PROPS,
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

    def _par_cor(data_new, data_old, cor_tgt, metric):
        for k in range(5, len(data_new) + 1):
            if eval_subset_correlation(data_new[:k], data_old, metric=metric) >= cor_tgt:
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

        ks_cor_par = pool.starmap(
            _par_cor,
            [(data_new, data_old, clu_tgt, metric) for prop, clu_tgt in zip(props, cors_tgt)]
        )
        ks_cor_par = [k / (len(data_old) * prop) for k, prop in zip(ks_cor_par, props)]

    return np.average(ks_clu_par), np.average(ks_cor_par)


def precompute_randnorm(
    data_old: List[Dict],
    random_seeds=10,
    metric="human",
    workers=10,
    props: List[float] = utils.PROPS,
) -> Tuple[List[float], List[float], float, float]:
    import subset2evaluate.select_subset

    clu_random = []
    cor_random = []
    for seed in range(random_seeds):
        clu_new, cor_new = eval_clucor(
            subset2evaluate.select_subset.basic(data_old, method="random", seed=seed),
            data_old,
            metric=metric,
            props=props
        )
        clu_random.append(clu_new)
        cor_random.append(cor_new)
    clu_random = np.average(clu_random, axis=0)
    cor_random = np.average(cor_random, axis=0)

    pars_clu_rand = []
    pars_cor_rand = []

    for seed in range(random_seeds, 2*random_seeds):
        par_clu_rand, par_cor_rand = eval_clucor_par(
            subset2evaluate.select_subset.basic(data_old, method="random", seed=seed),
            data_old,
            clu_random,
            cor_random,
            metric=metric,
            workers=workers,
            props=props,
        )
        pars_clu_rand.append(par_clu_rand)
        pars_cor_rand.append(par_cor_rand)

    return (clu_random, cor_random), (np.average(pars_clu_rand), np.average(pars_cor_rand))


def eval_clucor_par_randnorm(
    data_new: List[Dict],
    data_old: List[Dict],
    random_seeds=10,
    metric="human",
    clucor_precomputed=None,
    props: List[float] = utils.PROPS,
) -> Tuple[float, float]:

    if clucor_precomputed is not None:
        (clu_random, cor_random), (clu_random_norm, cor_random_norm) = clucor_precomputed
    else:
        (clu_random, cor_random), (clu_random_norm, cor_random_norm) = precompute_randnorm(
            data_old, random_seeds=random_seeds, metric=metric, props=props,
        )

    # compute the parity of the new data
    par_clu, par_cor = eval_clucor_par(
        data_new, data_old,
        clu_random, cor_random,
        metric=metric,
        props=props,
    )

    return par_clu/clu_random_norm, par_cor/cor_random_norm


def eval_subset_accuracy(data_new: List[Dict], data_old: List[Dict], metric="human"):
    # evaluates against ordering from data_old
    import itertools

    models = list(data_old[0]["scores"].keys())

    if type(metric) is tuple:
        metric1, metric2 = metric
    else:
        metric1 = metric
        metric2 = metric

    scores_new = get_model_absolute(data_new, metric=metric1)
    scores_old = get_model_absolute(data_old, metric=metric2)

    result = []
    for model1, model2 in itertools.combinations(models, 2):
        result.append((scores_old[model1] < scores_old[model2]) == (scores_new[model1] < scores_new[model2]))

    return np.average(result)


def eval_subset_correlation(data_new: List[Dict], data_old: List[Dict], metric="human"):
    # evaluates spearman correlation of systems
    import scipy.stats
    import warnings

    models = list(data_old[0]["scores"].keys())

    if type(metric) is tuple:
        metric1, metric2 = metric
    else:
        metric1 = metric
        metric2 = metric

    scores_new = get_model_absolute(data_new, metric=metric1)
    scores_old = get_model_absolute(data_old, metric=metric2)

    values_new = [scores_new[sys] for sys in models]
    values_old = [scores_old[sys] for sys in models]

    # handle constant input warning
    with warnings.catch_warnings(category=scipy.stats.ConstantInputWarning, action="error"):
        try:
            return scipy.stats.spearmanr(values_old, values_new).correlation
        except scipy.stats.ConstantInputWarning:
            return 0


def eval_subset_clusters(data: List[Dict], metric="human"):
    from scipy.stats import wilcoxon
    import warnings

    # if we have just 3 samples, we can't say that there are clusters
    if len(data) < 3:
        return 1

    # for compatibility with the rest of the code, but use metric1 anyway
    if type(metric) is tuple:
        metric1, metric2 = metric
    else:
        metric1 = metric
        metric2 = metric

    # sort from top
    model_ord = list(get_model_absolute(data, metric=metric1).items())
    model_ord.sort(key=lambda x: x[1], reverse=True)
    model_ord = [model for model, _ in model_ord]

    def get_scores(model):
        return [line["scores"][model][metric1] for line in data]

    clusters = [[get_scores(model_ord.pop(0))]]
    while model_ord:
        model_scores = get_scores(model_ord.pop(0))
        diffs = [x - y for x, y in zip(model_scores, clusters[-1][-1])]

        with warnings.catch_warnings(action="ignore"):
            if all([d == 0 for d in diffs]) or wilcoxon(diffs, alternative="less").pvalue < 0.05:
                clusters.append([model_scores])
            else:
                clusters[-1].append(model_scores)

    return len(clusters)


def get_model_absolute(data_new, metric="human") -> Dict[str, float]:
    import collections
    import numpy as np

    scores_new = collections.defaultdict(list)

    models = list(data_new[0]["scores"].keys())
    for line in data_new:
        for model in models:
            scores_new[model].append(line["scores"][model][metric])

    scores_new = {
        model: np.average(scores_new[model])
        for model in models
    }

    return scores_new


def get_model_ordering(data_new: List[Dict], metric="human") -> Dict[str, int]:
    scores_new = get_model_absolute(data_new, metric)

    # sort to get ordering
    scores_new = list(scores_new.items())
    # sort from highest
    scores_new.sort(key=lambda x: x[1], reverse=True)

    model_ord = {
        model: model_i
        for model_i, (model, model_v) in enumerate(scores_new)
    }

    return model_ord


def eval_order_accuracy(scores_new: Dict[str, float], scores_old: Dict[str, float]):
    # evaluates against ordering from data_old
    import itertools
    import numpy as np

    models = list(scores_old.keys())

    result = []
    for model1, model2 in itertools.combinations(models, 2):
        result.append((scores_old[model1] < scores_old[model2]) == (scores_new[model1] < scores_new[model2]))

    return np.average(result)


def eval_metrics_correlations(data: List[Dict], metric_target="human", display=False):
    import scipy.stats
    import collections
    metrics = set(list(data[0]["scores"].values())[0])
    data_y = {
        metric: [
            line["scores"][model][metric]
            for line in data
            for model in data[0]["scores"].keys()
        ]
        for metric in metrics
    }
    data_y_tgt = [
        line["scores"][model][metric_target]
        for line in data
        for model in data[0]["scores"].keys()
    ]
    corrs = {
        metric: scipy.stats.pearsonr(data_y[metric], data_y_tgt).correlation
        for metric in metrics
        if metric != metric_target
    }
    corrs = list(corrs.items())
    corrs.sort(key=lambda x: x[1], reverse=True)
    corrs = collections.OrderedDict(corrs)

    if display:
        for metric, cor in corrs.items():
            print(f"{metric:<40} {cor:.1%}")

    return corrs


def main_cli():
    import argparse

    args = argparse.ArgumentParser(
        description="Meta-evaluate subset selection methods with rank correlation and cluster count."
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

    clu_new, cor_new = eval_clucor(args.data_old, args.data_new, args.metric)

    print(f"Correlation: {np.average(cor_new):.1%}")
    print(f"Clusters: {np.average(clu_new):.2f}")
