from typing import Any, List, Tuple, Union
from functools import partial
import numpy as np
import subset2evaluate.evaluate
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def random_subset(data, seed=None, **kwargs) -> List[float]:
    import random
    r = random.Random(seed)
    return [r.random() for _ in data]


def metric_avg(data, metric, **kwargs) -> List[float]:
    return [
        -np.average([model_v[metric] for model_v in item["scores"].values()])
        for item in data
    ]


def metric_var(data, metric, **kwargs) -> List[float]:
    return [
        np.var([model_v[metric] for model_v in item["scores"].values()])
        for item in data
    ]


def pointwise_alignment(data, metric, metric_target=None, **kwargs) -> List[float]:
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


def pyirt(data, metric, return_model=False, load_model=None, model="4pl_score", dropout=0.25, epochs=1000, enforce_positive_disc=False, **kwargs) -> Union[List[float], Tuple[List[float], Any]]:
    try:
        import py_irt
    except ImportError as e:
        raise Exception("Please install py-irt with `pip install git+https://github.com/zouharvi/py-irt.git")
    
    import py_irt
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


def _assert_comet_version():
    try:
        import comet
    except ImportError as e:
        raise Exception("Please install COMET with `pip install git+https://github.com/zouharvi/PreCOMET.git`")

    import comet
    if "HypothesislessRegression" not in dir(comet.models):
        raise Exception("Please install COMET with `pip install git+https://github.com/zouharvi/PreCOMET.git`")


def precomet(data, model_path, return_model=False, load_model=None, reverse=False, **kwargs) -> Union[List, Tuple[List, Any]]:
    import os
    prev_tqdm_setting = os.environ.get("TQDM_DISABLE", None)
    os.environ["TQDM_DISABLE"] = "1"
    _assert_comet_version()

    import logging
    import comet
    import warnings

    logging.disable(logging.INFO)

    with warnings.catch_warnings(action="ignore"):
        if load_model is not None:
            model = load_model
        elif os.path.exists(model_path):
            model = comet.load_from_checkpoint(model_path)
        else:
            model = comet.load_from_checkpoint(comet.download_model(model_path))
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


def precomet_dual(data, model_path1, model_path2, return_model=False, load_model=None, reverse=False, **kwargs) -> Union[List, Tuple[List, Any]]:
    import os
    tqdm_disable_prev = os.environ.get("TQDM_DISABLE", None)
    os.environ["TQDM_DISABLE"] = "1"
    _assert_comet_version()

    import comet
    import warnings
    import logging

    logging.disable(logging.INFO)

    with warnings.catch_warnings(action="ignore"):
        if load_model is not None:
            model1, model2 = load_model
        else:
            if os.path.exists(model_path1):
                model1 = comet.load_from_checkpoint(model_path1)
            else:
                model1 = comet.load_from_checkpoint(comet.download_model(model_path1))

            if os.path.exists(model_path2):
                model2 = comet.load_from_checkpoint(model_path2)
            else:
                model2 = comet.load_from_checkpoint(comet.download_model(model_path2))
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
        return np.average(out)

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


METHODS = {
    "random": random_subset,
    "metric_avg": metric_avg,
    "metric_var": metric_var,
    "diversity_bleu": diversity_bleu,
    "diversity_chrf": diversity_chrf,
    "diversity_unigram": diversity_unigram,

    "pointwise_alignment": pointwise_alignment,

    "pyirt_diff": partial(pyirt, fn_utility="diff"),
    "pyirt_disc": partial(pyirt, fn_utility="disc"),
    "pyirt_diffdisc": partial(pyirt, fn_utility="diffdisc"),
    "pyirt_feas": partial(pyirt, fn_utility="feas"),
    "pyirt_fic": partial(pyirt, fn_utility="fisher_information_content"),
    "pyirt_experiment": partial(pyirt, fn_utility="experiment"),

    "precomet": partial(precomet, reverse=False),

    "precomet_pal": partial(precomet, model_path="../../PreCOMET/lightning_logs/pal/checkpoints/e4.ckpt", reverse=False),
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
}
