from typing import Any, Dict, List, Literal, Optional, Tuple, Union
from functools import partial
import numpy as np
import subset2evaluate.evaluate
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def random_subset(data: List[Dict], seed=None, **kwargs) -> List[float]:
    import random
    r = random.Random(seed)
    return [r.random() for _ in data]


def metric_avg(data: List[Dict], metric, **kwargs) -> List[float]:
    return [
        -np.average([model_v[metric] for model_v in item["scores"].values()])
        for item in data
    ]


def metric_var(data: List[Dict], metric, **kwargs) -> List[float]:
    return [
        np.var([model_v[metric] for model_v in item["scores"].values()])
        for item in data
    ]


def metric_consistency(data: List[Dict], metric, metric_target=None, **kwargs) -> List[float]:
    import scipy.stats
    import warnings
    if metric_target is None:
        metric_target = metric

    data_ord = subset2evaluate.evaluate.get_model_absolute(data, metric=metric_target)
    systems = data_ord.keys()
    data_ord = [data_ord[sys] for sys in systems]

    def _fn(item):
        sys_absolute = subset2evaluate.evaluate.get_model_absolute([item], metric=metric)
        sys_absolute = [sys_absolute[sys] for sys in systems]
        with warnings.catch_warnings(category=scipy.stats.ConstantInputWarning, action="ignore"):
            corr = scipy.stats.spearmanr(sys_absolute, data_ord).correlation
        if np.isnan(corr):
            return 0.0
        return corr

    return [
        _fn(x)
        for x in data
    ]


def kmeans(
    data: List[Dict],
    budget: float,
    load_model: Optional[object] = None,
    return_model: bool = False,
    features: Literal["src", "tgt_rand", "tgt_0"] = "src",
    **kwargs
) -> List[float]:
    import sklearn.cluster
    import sentence_transformers
    import warnings
    import random

    if load_model is not None:
        model_embd = load_model
    else:
        with warnings.catch_warnings(action="ignore", category=FutureWarning):
            model_embd = sentence_transformers.SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    if features == "src":
        data_x = [line["src"] for line in data]
    elif features == "tgt_rand":
        r_feature = random.Random(0)
        data_x = [r_feature.choice(list(line["tgt"].values())) for line in data]
    elif features == "tgt_0":
        data_x = [list(line["tgt"].values())[0] for line in data]
    else:
        raise Exception(f"Unknown feature type: {features}")

    data_embd = model_embd.encode(data_x)

    # baseline don't pick any item
    data_y = [0 for _ in data]

    kmeans = sklearn.cluster.KMeans(n_clusters=budget, random_state=0).fit(data_embd)
    data_new = []
    # get closest to cluster center
    for i, center in enumerate(kmeans.cluster_centers_):
        dist = np.linalg.norm(data_embd - center, axis=1)
        idx = np.argmin(dist)
        data_y[idx] = 1
        data_new += [data[idx]]*1
        # cluster_size = np.sum(kmeans.labels_ == i)
        # # data_new += [data[idx]]*cluster_size

    if return_model:
        return data_y, model_embd
    else:
        return data_y


def _fn_information_content(item_old, item_irt, data_irt) -> float:
    information = 0
    for theta in data_irt["models"].values():
        x1 = np.exp(item_irt["disc"] * (theta + item_irt["diff"]))
        x2 = np.exp(item_irt["disc"] * item_irt["diff"])
        x3 = np.exp(item_irt["disc"] * theta)
        information += (item_irt["disc"]**2) * x1 / (x2 + x3)**2
    return information


def fn_irt_utility(item_old, item_irt, data_irt, fn_utility) -> float:
    if fn_utility == "fisher_information_content":
        return _fn_information_content(item_old, item_irt, data_irt)
    elif fn_utility == "diff":
        return -item_irt["diff"]
    elif fn_utility == "disc":
        return -item_irt["disc"]
    elif fn_utility == "diffdisc":
        return item_irt["diff"] * item_irt["disc"]
    elif fn_utility == "feas":
        return item_irt["feas"]


def pyirt(  # noqa: C901
    data: List[Dict],
    metric: str,
    return_model: bool = False,
    load_model: Optional[object] = None,
    model: str = "4pl_score",
    dropout=0.25,
    epochs=1000,
    enforce_positive_disc: bool = False,
    **kwargs
) -> Union[List[float], Tuple[List[float], Any]]:
    try:
        import py_irt
    except ImportError:
        raise Exception("Please install py-irt with `pip install git+https://github.com/zouharvi/py-irt.git")

    import py_irt.config
    import py_irt.dataset
    import py_irt.io
    import py_irt.training
    import py_irt.models
    import py_irt.models.abstract_model
    import pandas as pd

    if model not in py_irt.models.abstract_model._IRT_REGISTRY:
        raise Exception("Please install py-irt with `pip install git+https://github.com/zouharvi/py-irt.git")

    models = list(data[0]["scores"].keys())

    if load_model is not None:
        data_irt = load_model
    else:
        # we need median binarization if we are not using 4pl_score model
        median = np.median([
            model_v[metric]
            for line in data
            for model_v in line["scores"].values()
        ])
        dataset = pd.DataFrame({
            "model": models,
            **{
                f"item_{line['i']}": [
                    line["scores"][model][metric]
                    if "_score" in model else
                    line["scores"][model][metric] >= median
                    for model in models
                ]
                for line in data
            }
        })

        embeddings = None
        if "amortized_" in model:
            import sentence_transformers
            embd_model = sentence_transformers.SentenceTransformer("paraphrase-MiniLM-L12-v2")
            embeddings = embd_model.encode([line["src"] for line in data])
            embeddings = {f"item_{line['i']}": emb.tolist() for line, emb in zip(data, embeddings)}
            del embd_model

        dataset = py_irt.dataset.Dataset.from_pandas(
            dataset,
            subject_column="model",
            item_columns=[f"item_{line['i']}" for line in data],
            embeddings=embeddings,
        )

        config = py_irt.config.IrtConfig(
            model_type=model,
            log_every=100,
            dropout=dropout,
            priors="hiearchical",
            seed=0,
            deterministic=True,
        )
        trainer = py_irt.training.IrtModelTrainer(
            config=config,
            data_path=None,
            dataset=dataset,
            verbose=False
        )
        trainer.train(epochs=epochs, device='cuda')

        params = trainer.best_params

        # this flipping should not affect the predictions
        if enforce_positive_disc and np.average(params["disc"]) < 0:
            params["disc"] = -np.array(params["disc"])
            params["ability"] = -np.array(params["ability"])
            params["diff"] = -np.array(params["diff"])

        # normalize naming
        if "lambdas" in params:
            params["feas"] = params.pop("lambdas")

        # TODO: cross-check make sure that we do the predictions as the models were trained
        if "feas" in params:
            # 3PL/4PL
            data_irt = {
                "models": {model: model_v for model, model_v in zip(models, params["ability"])},
                "items": [
                    {"disc": disc, "diff": diff, "feas": feas}
                    for disc, diff, feas in zip(
                        params["disc"],
                        params["diff"],
                        params["feas"],
                    )
                ]
            }
        elif "disc" in params:
            data_irt = {
                "models": {model: model_v for model, model_v in zip(models, params["ability"])},
                "items": [
                    {"disc": disc, "diff": diff}
                    for disc, diff in zip(
                        params["disc"],
                        params["diff"],
                    )
                ]
            }
        else:
            data_irt = {
                "models": {model: model_v for model, model_v in zip(models, params["ability"])},
                "items": [
                    {"diff": diff}
                    for diff in params["diff"]
                ]
            }

    scores = [
        fn_irt_utility(item_old, item_irt, data_irt, kwargs["fn_utility"])
        for item_old, item_irt in zip(data, data_irt["items"])
    ]

    if return_model:
        return scores, data_irt
    else:
        return scores


def precomet(
    data: List[Dict],
    model_path: str,
    return_model: bool = False,
    load_model: Optional[object] = None,
    reverse: bool = False,
    **kwargs
) -> Union[List, Tuple[List, Any]]:
    import os
    prev_tqdm_setting = os.environ.get("TQDM_DISABLE", None)
    os.environ["TQDM_DISABLE"] = "1"

    import logging
    import precomet
    import warnings

    logging.disable(logging.INFO)

    with warnings.catch_warnings(action="ignore"):
        if load_model is not None:
            model = load_model
        elif os.path.exists(model_path):
            model = precomet.load_from_checkpoint(model_path)
        else:
            model = precomet.load_from_checkpoint(precomet.download_model(model_path))
        scores = model.predict([
            {"src": line["src"]}
            for line in data
        ], progress_bar=False).scores
        if reverse:
            scores = [-x for x in scores]

    logging.disable(logging.NOTSET)
    if prev_tqdm_setting is not None:
        os.environ["TQDM_DISABLE"] = prev_tqdm_setting
    else:
        os.environ.pop("TQDM_DISABLE")

    if return_model:
        return scores, model
    else:
        return scores


def sentinel_src(
    data: List[Dict],
    model_path: str,
    return_model: str = False,
    load_model: Optional[object] = None,
    reverse: bool = False,
    **kwargs
):
    """
    Sentinel-SRC is a method based on "Guardians of the Machine Translation Meta-Evaluation: Sentinel Metrics Fall In!"
    Authors: Stefano Perrella, Lorenzo Proietti, Alessandro Scirè, Edoardo Barba, Roberto Navigli
    Link: https://aclanthology.org/2024.acl-long.856.pdf

    To use this method, you need to install sentinel_metric with:
    ```
    pip install git+https://github.com/SapienzaNLP/guardians-mt-eval.git
    ```
    """
    import sentinel_metric
    if load_model is not None:
        model = load_model
    elif os.path.exists(model_path):
        model = sentinel_metric.load_from_checkpoint(model_path)
    else:
        model = sentinel_metric.load_from_checkpoint(sentinel_metric.download_model(model_path))

    scores = model.predict([
        {"src": line["src"]}
        for line in data
    ], progress_bar=False).scores
    if reverse:
        scores = [-x for x in scores]

    if return_model:
        return scores, model
    else:
        return scores


def precomet_dual(
    data: List[Dict],
    model_path1: str,
    model_path2: str,
    return_model: bool = False,
    load_model: Optional[object] = None,
    reverse: bool = False,
    **kwargs
) -> Union[List, Tuple[List, Any]]:
    import os
    tqdm_disable_prev = os.environ.get("TQDM_DISABLE", None)
    os.environ["TQDM_DISABLE"] = "1"

    import precomet
    import warnings
    import logging

    logging.disable(logging.INFO)

    with warnings.catch_warnings(action="ignore"):
        if load_model is not None:
            model1, model2 = load_model
        else:
            if os.path.exists(model_path1):
                model1 = precomet.load_from_checkpoint(model_path1)
            else:
                model1 = precomet.load_from_checkpoint(precomet.download_model(model_path1))

            if os.path.exists(model_path2):
                model2 = precomet.load_from_checkpoint(model_path2)
            else:
                model2 = precomet.load_from_checkpoint(precomet.download_model(model_path2))
        scores1 = model1.predict([
            {"src": line["src"]}
            for line in data
        ], progress_bar=False).scores
        scores2 = model2.predict([
            {"src": line["src"]}
            for line in data
        ], progress_bar=False).scores

    if reverse:
        scores = [-s1 * s2 for s1, s2 in zip(scores1, scores2)]
    else:
        scores = [s1 * s2 for s1, s2 in zip(scores1, scores2)]

    logging.disable(logging.NOTSET)
    if tqdm_disable_prev is not None:
        os.environ["TQDM_DISABLE"] = tqdm_disable_prev
    else:
        os.environ.pop("TQDM_DISABLE")

    if return_model:
        return scores, (model1, model2)
    else:
        return scores


def diversity_unigram(data, **kwargs) -> List[float]:
    import itertools
    import collections

    def _f(line):
        out = []
        for text_a, text_b in itertools.combinations(line["tgt"].values(), 2):
            text_a = collections.Counter(text_a.split())
            text_b = collections.Counter(text_b.split())
            if text_a.total() == 0 or text_b.total() == 0:
                out.append(1)
            else:
                out.append(2 * (text_a & text_b).total() / (text_a.total() + text_b.total()))
        return 0 if len(out) == 0 else np.average(out)

    # we prefer smallest similarity so flip
    return [
        -_f(line)
        for line in data
    ]


def diversity_bleu(data, **kwargs) -> List[float]:
    import itertools
    import sacrebleu
    metric = sacrebleu.metrics.BLEU(effective_order=True)

    def _f(line):
        if len(line["tgt"]) < 2:
            return 0
        return np.average([
            metric.sentence_score(
                text_a,
                [text_b],
            ).score
            for text_a, text_b in itertools.product(line["tgt"].values(), line["tgt"].values())
        ])

    # we prefer smallest similarity so flip
    return [
        -_f(line)
        for line in data
    ]


def diversity_chrf(data, **kwargs) -> List[float]:
    import itertools
    import sacrebleu
    metric = sacrebleu.metrics.CHRF()

    def _f(line):
        return np.average([
            metric.sentence_score(
                text_a,
                [text_b],
            ).score
            for text_a, text_b in itertools.product(line["tgt"].values(), line["tgt"].values())
        ])

    # we prefer smallest similarity so flip
    return [
        -_f(line)
        for line in data
    ]


def diversity_lm(data, **kwargs) -> List[float]:
    import itertools
    import sentence_transformers
    import warnings

    with warnings.catch_warnings(action="ignore", category=FutureWarning):
        model_embd = sentence_transformers.SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    embds = list(model_embd.encode([
        tgt
        for line in data
        for tgt in line["tgt"].values()
    ]))

    def _f(embds):
        return np.average([
            np.dot(embd_a, embd_b)
            for embd_a, embd_b in itertools.combinations(embds, 2)
        ])

    # we prefer smallest similarity so flip
    return [
        -_f([embds.pop(0) for _ in line["tgt"].values()])
        for line in data
    ]


def diversity(data, metric, **kwargs) -> List[float]:
    if metric.lower() == "bleu":
        return diversity_bleu(data, **kwargs)
    elif metric.lower() == "chrf":
        return diversity_chrf(data, **kwargs)
    elif metric.lower() == "unigram":
        return diversity_unigram(data, **kwargs)
    elif metric.lower() == "lm":
        return diversity_lm(data, **kwargs)
    else:
        raise Exception(f"Unknown diversity metric: {metric}")


def diffuse(
    data: List[Dict],
    budget: int,
    load_model=None,
    return_model=False,
    **kwargs
) -> List[float]:
    """
    DiffUse is a method based on "Label-Efficient Model Selection for Text Generation"
    Authors: Shir Ashury-Tahan, Ariel Gera, Benjamin Sznajder, Leshem Choshen, Liat Ein-Dor, Eyal Shnarch
    Link: https://aclanthology.org/2024.acl-long.456.pdf

    This method requires budget to be specified and can't return a prioritized list because
    the results non-sequentially depend on the budget parameter.
    """
    import warnings
    import sentence_transformers
    import itertools
    import scipy.cluster
    import copy

    if load_model is not None:
        # loading data and not model
        data_x, clusters_desc_all = load_model
    else:
        with warnings.catch_warnings(action="ignore", category=FutureWarning):
            model_embd = sentence_transformers.SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

        # encode all in batch
        data_x_all = [x for line in data for x in line["tgt"].values()]
        data_embd = list(model_embd.encode(data_x_all))

        # reconstruct original_structure
        data_x = []
        for line in data:
            data_x.append([data_embd.pop(0) for _ in line["tgt"]])
        assert len(data_embd) == 0

        # TODO: try using abs
        # compute average difference
        data_x = [
            np.average([
                embd_a - embd_b
                for embd_a, embd_b in itertools.combinations(line, 2)
            ], axis=1)
            for line in data_x
        ]
        clusters = scipy.cluster.hierarchy.linkage(data_x, method="ward")

        # maps cluster index to item sets
        clusters_map = [frozenset({i}) for i in range(len(data))]
        # shows sets of items in each cluster at each timepoint
        # start with each item being a cluster on its own
        clusters_desc_all = [{frozenset({i}) for i in range(len(data))}]
        for i in range(len(clusters)):
            # retrieve indices that are being merged
            cluster_0 = clusters_map[int(clusters[i, 0])]
            cluster_1 = clusters_map[int(clusters[i, 1])]

            # add the new cluster to our map
            cluster_new = cluster_0.union(cluster_1)
            clusters_map.append(cluster_new)

            cluster_desc = copy.deepcopy(clusters_desc_all[-1])
            # remove individual old clusters
            cluster_desc.remove(cluster_0)
            cluster_desc.remove(cluster_1)
            # add new cluster
            cluster_desc.add(cluster_new)
            clusters_desc_all.append(cluster_desc)

    # take first that fits within budget
    cluster_desc = [cluster_desc for cluster_desc in clusters_desc_all if len(cluster_desc) <= budget][0]

    # for each cluster compute the average distance
    cluster_desc = [list(fs) for fs in cluster_desc]
    # baseline don't pick any item
    data_y = [0 for _ in data]
    # get closest to cluster center
    for cluster_is in cluster_desc:
        data_embd_local = [data_x[i] for i in cluster_is]
        center = np.average(data_embd_local, axis=0)

        dist = np.linalg.norm(data_embd_local - center, axis=1)
        idx = np.argmin(dist)
        idx = cluster_is[idx]

        # set utility
        data_y[idx] = 1

    if return_model:
        return data_y, (data_x, clusters_desc_all)
    else:
        return data_y


def combinator(
    data: List[Dict],
    methods: List[Dict],
    operation: Literal["mul", "sum"] = "mul",
    normalize_zscore: bool = True,
    normalize_01: bool = True,
    **kwargs
) -> List[float]:
    scores = []
    for method_kwargs in methods:
        assert \
            "return_model" not in method_kwargs or not method_kwargs["return_model"], \
            "Cannot return model in combinator"
        scores.append(METHODS[method_kwargs["method"]](data, **method_kwargs))

    scores = np.array(scores)

    if normalize_zscore:
        # z-score
        scores = (scores - np.mean(scores, axis=1, keepdims=True)) / np.std(scores, axis=1, keepdims=True)

    if normalize_01:
        # make positive and in [0, 1]
        scores = (
            (scores - np.min(scores, axis=1, keepdims=True)) /
            (np.max(scores, axis=1, keepdims=True) - np.min(scores, axis=1, keepdims=True))
        )

    if operation == "mul":
        scores = np.prod(scores, axis=0)
    elif operation == "sum":
        scores = np.sum(scores, axis=0)
    else:
        raise Exception(f"Unknown operation: {operation}")

    return scores


METHODS = {
    "random": random_subset,
    "metric_avg": metric_avg,
    "metric_var": metric_var,
    "metric_cons": metric_consistency,

    "diversity": diversity,

    "combinator": combinator,

    "kmeans": kmeans,
    "diffuse": diffuse,

    "pyirt_diff": partial(pyirt, fn_utility="diff"),
    "pyirt_disc": partial(pyirt, fn_utility="disc"),
    "pyirt_diffdisc": partial(pyirt, fn_utility="diffdisc"),
    "pyirt_feas": partial(pyirt, fn_utility="feas"),
    "pyirt_fic": partial(pyirt, fn_utility="fisher_information_content"),
    "pyirt_experiment": partial(pyirt, fn_utility="experiment"),

    "precomet": partial(precomet, reverse=False),
    "precomet_var": partial(precomet, model_path="zouharvi/PreCOMET-var", reverse=True),
    "precomet_avg": partial(precomet, model_path="zouharvi/PreCOMET-avg", reverse=True),
    "precomet_diversity": partial(precomet, model_path="zouharvi/PreCOMET-diversity", reverse=True),

    "precomet_diff": partial(precomet, model_path="zouharvi/PreCOMET-diff", reverse=False),
    "precomet_disc": partial(precomet, model_path="zouharvi/PreCOMET-disc", reverse=True),
    "precomet_diffdisc_direct": partial(precomet, model_path="zouharvi/PreCOMET-diffdisc_direct", reverse=False),
    "precomet_diffdisc": partial(
        precomet_dual,
        model_path1="zouharvi/PreCOMET-diff",
        model_path2="zouharvi/PreCOMET-disc",
        reverse=False,
    ),
    "precomet_cons": partial(precomet, model_path="zouharvi/PreCOMET-cons", reverse=False),

    "sentinel_src": partial(sentinel_src, model_path="sapienzanlp/sentinel-src-da", reverse=True),
    "sentinel_src_mqm": partial(sentinel_src, model_path="sapienzanlp/sentinel-src-mqm", reverse=True),
}

METHOD_NAMES = {
    "random": "Random",
    "metric_avg": "MetricAvg",
    "metric_var": "MetricVar",
    "metric_cons": "MetricCons",
    "diversity": "Diversity",
    "pyirt_diffdisc": "DiffDisc",
}
