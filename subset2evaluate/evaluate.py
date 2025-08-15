from typing import Dict, List, Literal, Tuple, Union
import numpy as np
import subset2evaluate.utils as utils


def eval_spa(
        data_new: List[Dict],
        data_old: List[Dict],
        metric="human",
        props: List[float] = utils.PROPS
) -> List[float]:
    # both list or descriptor is fine
    data_new = utils.load_data(data_new)
    data_old = utils.load_data(data_old)

    spa_new = []
    for prop in props:
        k = int(len(data_old) * prop)
        spa_new.append(eval_subset_spa(data_new[:k], data_old, metric=metric))

    return spa_new


def eval_clucor(
        data_new: List[Dict],
        data_old: List[Dict],
        metric="human",
        props: List[float] = utils.PROPS
) -> Tuple[List[float], List[float]]:
    # both list or descriptor is fine
    data_new = utils.load_data(data_new)
    data_old = utils.load_data(data_old)

    clu_new = []
    cor_new = []
    for prop in props:
        k = int(len(data_old) * prop)
        clu_new.append(eval_subset_clusters(data_new[:k], metric=metric))
        cor_new.append(eval_subset_correlation(
            data_new[:k], data_old, metric=metric))

    return clu_new, cor_new


def eval_clucor_par(
    data_new: List[Dict],
    data_old: List[Dict],
    clus_tgt: List[float],
    cors_tgt: List[float],
    metric="human",
    props: List[float] = utils.PROPS,
    workers=10,
) -> Tuple[List[float], List[float]]:
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
            [(data_new, clu_tgt, metric)
             for prop, clu_tgt in zip(props, clus_tgt)]
        )
        ks_clu_par = [k / (len(data_old) * prop)
                      for k, prop in zip(ks_clu_par, props)]

        ks_cor_par = pool.starmap(
            _par_cor,
            [(data_new, data_old, clu_tgt, metric)
             for prop, clu_tgt in zip(props, cors_tgt)]
        )
        ks_cor_par = [k / (len(data_old) * prop)
                      for k, prop in zip(ks_cor_par, props)]

    return np.average(ks_clu_par), np.average(ks_cor_par)


def eval_spa_par(
    data_new: List[Dict],
    data_old: List[Dict],
    spas_tgt: List[float],
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

    def _par_spa(data_new, clu_tgt, metric):
        for k in range(5, len(data_new) + 1):
            if eval_subset_spa(data_new[:k], data_old, metric=metric) >= clu_tgt:
                break
        return k

    # multiprocess for each prop rather than k because the thread
    # orchestration would be more expensive otherwise
    with multiprocessing.pool.ThreadPool(min(workers, len(props))) as pool:
        ks_spa_par = pool.starmap(
            _par_spa,
            [(data_new, clu_tgt, metric)
             for prop, clu_tgt in zip(props, spas_tgt)]
        )
        ks_spa_par = [k / (len(data_old) * prop)
                      for k, prop in zip(ks_spa_par, props)]

    return np.average(ks_spa_par)


def precompute_clucor_randnorm(
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
            subset2evaluate.select_subset.basic(
                data_old, method="random", seed=seed),
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
            subset2evaluate.select_subset.basic(
                data_old, method="random", seed=seed),
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
        (clu_random, cor_random), (clu_random_norm,
                                   cor_random_norm) = clucor_precomputed
    else:
        (clu_random, cor_random), (clu_random_norm, cor_random_norm) = precompute_clucor_randnorm(
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


def precompute_spa_randnorm(
    data_old: List[Dict],
    random_seeds=10,
    metric="human",
    workers=10,
    props: List[float] = utils.PROPS,
) -> Tuple[List[float], List[float], float, float]:
    import subset2evaluate.select_subset

    spa_random = []
    for seed in range(random_seeds):
        spa_new = eval_spa(
            subset2evaluate.select_subset.basic(
                data_old, method="random", seed=seed),
            data_old,
            metric=metric,
            props=props
        )
        spa_random.append(spa_new)
    spa_random = np.average(spa_random, axis=0)

    pars_spa_rand = []

    for seed in range(random_seeds, 2*random_seeds):
        par_spa_rand = eval_spa_par(
            subset2evaluate.select_subset.basic(
                data_old, method="random", seed=seed),
            data_old,
            spa_random,
            metric=metric,
            workers=workers,
            props=props,
        )
        pars_spa_rand.append(par_spa_rand)

    return spa_random, np.average(pars_spa_rand)


def eval_spa_par_randnorm(
    data_new: List[Dict],
    data_old: List[Dict],
    random_seeds=10,
    metric="human",
    spa_precomputed=None,
    props: List[float] = utils.PROPS,
) -> Tuple[float, float]:

    if spa_precomputed is not None:
        spa_random, spa_random_norm = spa_precomputed
    else:
        spa_random, spa_random_norm = precompute_spa_randnorm(
            data_old, random_seeds=random_seeds, metric=metric, props=props,
        )

    # compute the parity of the new data
    par_spa = eval_spa_par(
        data_new, data_old,
        spa_random,
        metric=metric,
        props=props,
    )

    return par_spa/spa_random_norm


def eval_subset_pairwise_accuracy(data_new: List[Dict], data_old: List[Dict], metric="human"):
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
        result.append((scores_old[model1] < scores_old[model2]) == (
            scores_new[model1] < scores_new[model2]))

    return np.average(result)


def compute_pairwise_p_values(seg_scores, num_permutations=1000):
    """
    Author: Brian Thompson
    Date: June 2024

    Suppose we have test set consisting of L=5 segments, and two systems, systemsA and systemB,
    for which we have segment-level scores scoresA and scoresB:
       scoresA = [0.8, 0.9, 0.7, 1.0, 0.6]
       scoresB = [0.2, 0.3, 0.1, 0.4, 0.0]

    Typically we would average segment-level scores to get system level scores, but for convenience later on
    we will define system scores to be the sum of segment-level scores. This gives us a delta system-level score of:
        test_delta = sum(scoresA) - sum(scoresB) = 4.0 - 1.0 = 3.0

    To run a paired permutation test, we first generate a new set of scores scores0,
    where each score0[i] is randomly selected from either scoresA[i] or scoresB[i].
    Let's define a random boolean mask:
       m = [1, 0, 0, 1, 1]

    and used it to select scores0:
       scores0 = m.*scoresA + (1-m).*scoresB = [0.8, 0.3, 0.1, 1.0, 0.6]   # selected from [A, B, B, A, A], respectively

    Likewise, we compose scores1 using all the scores which were not selected for scores0:
       scores1 = (1-m).*scoresA + m.*scoresB = [0.2, 0.9, 0.7, 0.4, 0.0]   # selected from [B, A, A, B, B], respectively

    To get the delta system-level score for our two mock systems, we need to compute:
       null_delta = sum(scores0) - sum(scores1)
                  = sum(m.*scoresA + (1-m).*scoresB) - sum((1-m).*scoresA + m.*scoresB)
                  = sum((2m-1).*scoresA) - sum((2m-1).*scoresB
                  = (2m-1) * scoresA.T - (2m-1) * scoresB.T
                  = [ 1, -1, -1,  1,  1] * [[0.8],  -  [ 1, -1, -1,  1,  1] * [[0.2],  =  0.8 - 0.2  =  0.6
                                            [0.9],                             [0.3],
                                            [0.7],                             [0.1],
                                            [1.0],                             [0.4],
                                            [0.6]]                             [0.0]]

    To compute many different permutations, we replace the vector m with a matrix of size (num_permutations, L):
       null_delta = [[ 1,  1, -1, -1, -1], * [[0.8],  -  [[ 1,  1, -1, -1, -1], * [[0.2],  = [[-0.6],  - [[ 0.0],   =  [[-0.6]
                     [ 1, -1,  1, -1,  1],    [0.9],      [ 1, -1,  1, -1,  1],    [0.3],     [ 0.2],     [-0.4],       [ 0.6],
                     [ 1, -1,  1,  1, -1],    [0.7],      [ 1, -1,  1,  1, -1],    [0.1],     [ 1.0],     [ 0.4],       [ 0.6],
                     [-1,  1, -1, -1,  1],    [1.0],      [-1,  1, -1, -1,  1],    [0.4],     [-1.0],     [-0.4],       [-0.6],
                     [ 1,  1,  1, -1,  1],    [0.6]]      [ 1,  1,  1, -1,  1],    [0.0]]     [ 2.0],     [ 0.2],       [ 1.8],
                     [-1,  1, -1,  1, -1],                [-1,  1, -1,  1, -1],               [-0.2],     [ 0.4],       [-0.6],
                     [ 1,  1,  1,  1,  1],                [ 1,  1,  1,  1,  1],               [ 4.0],     [ 1.0],       [ 3.0],
                     [ 1, -1,  1, -1,  1],                [ 1, -1,  1, -1,  1],               [ 0.2],     [-0.4],       [ 0.6],
                     [ 1,  1, -1, -1,  1],                [ 1,  1, -1, -1,  1],               [ 0.6],     [ 0.0],       [ 0.6],
                     [-1,  1, -1, -1, -1]]                [ 1, -1, -1,  1, -1]]               [-2.2]]     [-0.4]]       [-1.8]]

    To test the significance that system A is better than system B, we compute:
       null_delta >= test_delta  =  [[-0.6]  >= 3   =   [[False],
                                     [ 0.6],             [False],
                                     [ 0.6],             [False],
                                     [-0.6],             [False],
                                     [ 1.8],             [False],
                                     [-0.6],             [False],
                                     [ 3.0],             [True ],
                                     [ 0.6],             [False],
                                     [ 0.6],             [False],
                                     [-1.8]]             [False]]

    The p value is the fraction of the time that null_delta >= test_delta, in this case 1/10 = 0.1

    The above discussion was for a single system pair, but we actually need to compute p values for each pairwise
    within a set systems systemA, systemB, ... systemN. In practice, the computation bottleneck is generating
    the random boolean vector m, so we generate m once and use it for all pairs of systems.

    Reusing m also allows us to avoid most of the N^2 computations by pre-computing (2m-1) * scoresA.T,
    (2m-1) * scoresB.T, ..., (2m-1) * scoresN.T.

    Test speed:
    python -m timeit -s "import numpy as np; from pairwise_paired_permutation_test import compute_pairwise_p_values; x=np.random.random(size=(14,1300))" "compute_pairwise_p_values(x, num_permutations=1000)"

    :param seg_scores: segment-level scores, with shape (num_systems, num_segments)
    :param num_permutations: Number of permutations for permutation test
    :return: np.array of size (num_systems, num_systems), where the upper triangle has been populated
       with p-values for the hypothesis that system[i] > system[j]
    """
    import numpy as np
    num_systems, num_segments = seg_scores.shape

    rng = np.random.default_rng()
    # initialize in range [0, 1)
    two_m_minus_one = rng.random(
        size=(num_permutations, num_segments), dtype=np.float32)
    # quantize to 0 or 1, in place
    np.rint(two_m_minus_one, out=two_m_minus_one, casting='same_kind')
    # scale and shift to get -1.0 and +1.0, in place
    two_m_minus_one *= 2.0
    two_m_minus_one -= 1.0

    # shape: (num_systems, num_segments)
    seg_scores = seg_scores.astype(np.float32)
    sys_scores = np.sum(seg_scores, axis=1)  # shape: (num_systems, )

    # shape: (num_permutations, num_systems)
    partial = np.matmul(two_m_minus_one, seg_scores.T)

    # initialize p value matrix to NaN
    p_vals = np.empty((num_systems, num_systems,)) * np.nan
    # populate upper triangle
    for ii in range(num_systems):
        for jj in range(ii + 1, num_systems):
            # shape: (num_permutations, )
            null_delta = partial[:, ii] - partial[:, jj]
            test_delta = sys_scores[ii] - sys_scores[jj]  # float
            p_vals[ii, jj] = np.sum(
                null_delta >= test_delta) / num_permutations

    return p_vals


def compute_one_minus_pce(metric_pairwise_p_vals, human_pairwise_p_vals):
    """
    Author: Brian Thompson
    Date: June 2024

    Pairwise Confidence Error (PCE) is the absolute difference between
      the p value for the conclusion that one system is better than another given human judgements and
      the p value for the conclusion for the same system comparison given metric judgements,
      averaged over all system pairings for a set of systems.

    We return 1-PCE to be comparable with pairwise accuracy [i.e. range from 0 to 1, higher is better]

    :param human_pairwise_p_vals: np.array of shape (num_systems, num_systems),
        where the upper triangle has been populated with p-values for system[i] > system[j]
        computed from human judgements
    :param metric_pairwise_p_vals: np.array of shape (num_systems, num_systems),
        where the opper triangle has been populated with p-values for system[i] > system[j]
        computed from metric scores
    :return: 1-PCE
    """
    num_systems = human_pairwise_p_vals.shape[0]
    upper_tri_idxs = np.triu_indices(num_systems, 1)
    return 1.0 - np.mean(np.abs(human_pairwise_p_vals - metric_pairwise_p_vals)[upper_tri_idxs])


def eval_subset_spa(data_new: List[Dict], data_old: List[Dict], metric="human"):
    # compute soft pairwise accuracy
    import numpy as np

    models = list(data_old[0]["scores"].keys())

    if type(metric) is tuple:
        metric1, metric2 = metric
    else:
        metric1 = metric
        metric2 = metric

    values_new = [[line["scores"][model][metric1]
                   for model in models] for line in data_new]
    values_old = [[line["scores"][model][metric2]
                   for model in models] for line in data_old]

    # transpose to be in (n_models, n_samples) format
    values_old = np.array(values_old).T
    values_new = np.array(values_new).T
    pvals_old = compute_pairwise_p_values(values_old)
    pvals_new = compute_pairwise_p_values(values_new)

    return compute_one_minus_pce(pvals_new, pvals_old)


def eval_subset_top(data_new: List[Dict], data_old: List[Dict], metric="human"):
    if type(metric) is tuple:
        metric1, metric2 = metric
    else:
        metric1 = metric
        metric2 = metric

    scores_new: Dict[str, float] = get_model_absolute(data_new, metric=metric1)
    scores_old: Dict[str, float] = get_model_absolute(data_old, metric=metric2)

    # compute name of top system in scores_new
    top_new = max(scores_new, key=scores_new.get)
    # compute rank of top_new in scores_old
    scores_old_sorted = sorted(
        scores_old.keys(), key=scores_old.get, reverse=True)
    rank_top_new = scores_old_sorted.index(top_new)

    # turn into percentile rank
    return 1 - (rank_top_new / (len(scores_old) - 1))


def eval_subset_correlation(
    data_new: List[Dict],
    data_old: List[Dict],
    metric="human",
    correlation: Literal["kendall", "spearman", "pearson"] = "kendall",
):
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
            if correlation == "spearman":
                return scipy.stats.spearmanr(values_old, values_new).correlation
            elif correlation == "pearson":
                return scipy.stats.pearsonr(values_old, values_new).correlation
            elif correlation == "kendall":
                return scipy.stats.kendalltau(values_old, values_new, variant="b").correlation
            else:
                raise ValueError(f"Unknown correlation type: {correlation}.")
        except scipy.stats.ConstantInputWarning:
            return 0


def eval_subset_error(
    data_new: List[Dict],
    data_old: List[Dict],
    metric="human",
    error: Literal["absolute", "squared", "root_squared"] = "absolute",
):
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

    if error == "absolute":
        return np.average([abs(x - y) for x, y in zip(values_old, values_new)])
    elif error == "squared":
        return np.average([(x - y) ** 2 for x, y in zip(values_old, values_new)])
    elif error == "root_squared":
        return np.sqrt(np.average([(x - y) ** 2 for x, y in zip(values_old, values_new)]))
    else:
        raise ValueError(f"Unknown error type: {error}.")


def eval_subset_clusters(data: List[Dict], metric="human"):
    return len(compute_clusters(data, metric=metric))


def eval_subset_clusters_top(
    data_new: List[Dict],
    data_old: List[Dict],
    clusters_old: Union[None, List[List[str]]] = None,
    metric="human"
):
    clusters_new = compute_clusters(data_new, metric=metric)
    if clusters_old is None:
        clusters_old = compute_clusters(data_old, metric=metric)
    # compare top clusters
    return 2*len(set(clusters_new[0]) & set(clusters_old[0])) / (len(clusters_new[0]) + len(clusters_old[0]))


def compute_clusters(data: List[Dict], metric="human"):
    from scipy.stats import wilcoxon
    import warnings

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

    # if we have just 3 samples, we can't say that there are clusters, so everything is in one cluster
    if len(data) < 3:
        return [model_ord]

    def get_scores(model):
        return [line["scores"][model][metric1] for line in data]

    model_first = model_ord.pop(0)
    clusters = [[(model_first, get_scores(model_first))]]
    while model_ord:
        model = model_ord.pop(0)
        model_scores = get_scores(model)
        diffs = [x - y for x, y in zip(model_scores, clusters[-1][-1][1])]

        with warnings.catch_warnings(action="ignore"):
            # TODO: is the diff clause correct?
            if all([d == 0 for d in diffs]) or wilcoxon(diffs, alternative="less").pvalue < 0.05:
                clusters.append([(model, model_scores)])
            else:
                clusters[-1].append((model, model_scores))

    return [[model for model, model_score in cluster] for cluster in clusters]


def compute_clusters_pvalues(data: List[Dict], metric="human") -> List[float]:
    # TODO: merge with compute_clusters to provide both at the same time
    from scipy.stats import wilcoxon
    import warnings

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

    # if we have just 3 samples, we can't say that there are clusters, so everything is in one cluster
    if len(data) < 3:
        return []

    def get_scores(model):
        return [line["scores"][model][metric1] for line in data]

    model_first = model_ord.pop(0)
    clusters = [[(model_first, get_scores(model_first))]]
    pvalues = []
    while model_ord:
        model = model_ord.pop(0)
        model_scores = get_scores(model)
        diffs = [x - y for x, y in zip(model_scores, clusters[-1][-1][1])]

        with warnings.catch_warnings(action="ignore"):
            if all([d == 0 for d in diffs]):
                pvalue = 1
            else:
                pvalue = wilcoxon(diffs, alternative="less").pvalue
            pvalues.append(pvalue)

            if pvalue < 0.05:
                clusters.append([(model, model_scores)])
            else:
                clusters[-1].append((model, model_scores))

    return pvalues


def compute_clusters_soft(data: List[Dict], metric="human"):
    from scipy.stats import wilcoxon
    import warnings

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

    # if we have just 3 samples, we can't say that there are clusters, so everything is in one cluster
    if len(data) < 3:
        return [model_ord]

    def get_scores(model):
        return [line["scores"][model][metric1] for line in data]

    model_first = model_ord.pop(0)
    clusters = [[(model_first, get_scores(model_first))]]
    pvalues = []
    while model_ord:
        model = model_ord.pop(0)
        model_scores = get_scores(model)
        diffs = [x - y for x, y in zip(model_scores, clusters[-1][-1][1])]

        with warnings.catch_warnings(action="ignore"):
            if all([d == 0 for d in diffs]):
                pvalue = 1
            else:
                pvalue = wilcoxon(diffs, alternative="less").pvalue
            pvalues.append(pvalue)

            if pvalue < 0.05:
                clusters.append([(model, model_scores)])
            else:
                clusters[-1].append((model, model_scores))

    return pvalues


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
        result.append((scores_old[model1] < scores_old[model2]) == (
            scores_new[model1] < scores_new[model2]))

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
