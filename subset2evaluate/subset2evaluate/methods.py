from typing import Callable
import numpy as np
import irt_mt_dev.utils as utils
from functools import partial
import random
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def random_subset(data, seed=None, **kwargs):
    import random
    random.Random(seed).shuffle(data)
    return data


def metric_avg(data, metric, **kwargs):
    data.sort(key=lambda item: np.average(
        [sys_v[metric] for sys_v in item["scores"].values()]
    ))
    return data


def metric_var(data, metric, **kwargs):
    data.sort(key=lambda item: np.var(
        [sys_v[metric] for sys_v in item["scores"].values()]
    ), reverse=True)
    return data



def _fn_information_content_old(item_irt, data_irt):
    # This formula is based on the simplified formula of Rodriquez et al 2021
    information = 0
    for theta in data_irt["systems"].values():
        prob = utils.pred_irt(
            theta,
            item_irt
        )
        information += prob*(1-prob)*(item_irt["disc"]**2)
    return information

def _fn_information_content(item_old, item_irt, data_irt):
    information = 0
    for theta in data_irt["systems"].values():
        x1 = np.exp(item_irt["disc"]*(theta+item_irt["diff"]))
        x2 = np.exp(item_irt["disc"]*item_irt["diff"])
        x3 = np.exp(item_irt["disc"]*theta)
        information += (item_irt["disc"]**2)*x1/(x2+x3)**2
    return information

def _fn_experiment(item_old, item_irt, data_irt):
    information = 0
    for theta in data_irt["systems"].values():
        x1 = np.exp(item_irt["disc"]*(theta+item_irt["diff"]))
        x2 = np.exp(item_irt["disc"]*item_irt["diff"])
        x3 = np.exp(item_irt["disc"]*theta)
        information += (item_irt["disc"]**2)*x1/(x2+x3)**2
    return information

def fn_irt_utility(item_old, item_irt, data_irt, fn_utility):
    if fn_utility == "experiment":
        return _fn_experiment(item_old, item_irt, data_irt)
    elif fn_utility == "fisher_information_content":
        return _fn_information_content(item_old, item_irt, data_irt)
    elif fn_utility == "diff":
        return -item_irt["diff"]
    elif fn_utility == "disc":
        return -item_irt["disc"]
    elif fn_utility == "diffdisc":
        return item_irt["diff"]*item_irt["disc"]
    elif fn_utility == "feas":
        return item_irt["feas"]

def pyirt(data, metric, return_model=False, load_model=None, model="4pl_score", dropout=0.25, epochs=1000, enforce_positive_disc=False, **kwargs):
    import py_irt
    import py_irt.config
    import py_irt.dataset
    import py_irt.io
    import py_irt.training
    import pandas as pd


    systems = list(data[0]["scores"].keys())

    if load_model is not None:
        data_irt = load_model
    else:
        # we need median binarization if we are not using 4pl_score model
        median = np.median([
            system_v[metric]
            for line in data
            for system_v in line["scores"].values()
        ])
        dataset = pd.DataFrame({
            "system": systems,
            **{
                f"item_{line['i']}": [
                    line["scores"][system][metric]
                    if "_score" in model else
                    line["scores"][system][metric] >= median
                    for system in systems
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
            subject_column="system",
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
        # 3PL/4PL
        if "feas" in params:
            data_irt = {
                "systems": {sys: sys_v for sys, sys_v in zip(systems, params["ability"])},
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
                "systems": {sys: sys_v for sys, sys_v in zip(systems, params["ability"])},
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
                "systems": {sys: sys_v for sys, sys_v in zip(systems, params["ability"])},
                "items": [
                    {"diff": diff}
                    for diff in params["diff"]
                ]
            }

    items_joint = list(zip(data, data_irt["items"]))
    items_joint.sort(
        key=lambda x: fn_irt_utility(x[0], x[1], data_irt, kwargs["fn_utility"]),
        reverse=True
    )

    items = [x[0] for x in items_joint]

    if return_model:
        return items, data_irt
    else:
        return items


def premlp_other(data, data_train, fn_utility: Callable, **kwargs):
    # turn off warnings from sentence-transformers
    import warnings
    warnings.filterwarnings('ignore')
    import sentence_transformers
    import sklearn.neural_network

    embd_model = sentence_transformers.SentenceTransformer("paraphrase-MiniLM-L12-v2")
    data_x_train = embd_model.encode([line["src"] for line in data_train])
    data_y_train = [fn_utility(line) for line in data_train]

    model = sklearn.neural_network.MLPRegressor(
        hidden_layer_sizes=(128, 16),
        max_iter=1000,
        verbose=False,
    )
    model.fit(data_x_train, data_y_train)
    data_x_test = embd_model.encode([line["src"] for line in data])
    data_y_test = model.predict(data_x_test)

    data_w_score = list(zip(data, data_y_test))
    data_w_score.sort(key=lambda x: x[1], reverse=True)

    return [x[0] for x in data_w_score]

def premlp_irt(data, data_train, load_model=None, return_model=False, **kwargs):
    import sklearn.neural_network
    import sentence_transformers

    # turn off warnings from sentence-transformers
    import warnings
    warnings.filterwarnings('ignore')

    embd_model = sentence_transformers.SentenceTransformer("paraphrase-MiniLM-L12-v2")
    data_x_train = embd_model.encode([line["src"] for line in data_train])

    model_fn = lambda: sklearn.neural_network.MLPRegressor(
        hidden_layer_sizes=(128, 16),
        max_iter=1000,
        verbose=False,
    )

    if load_model is not None:
        model_diff, model_disc = load_model
    else:
        model_diff = model_fn()
        model_disc = model_fn()

        data_y_diff = [line["irt"]["diff"] for line in data_train]
        data_y_disc = [line["irt"]["disc"] for line in data_train]

        model_diff.fit(data_x_train, data_y_diff)
        model_disc.fit(data_x_train, data_y_disc)

    data_x_test = embd_model.encode([line["src"] for line in data])
    data_y_diff = model_diff.predict(data_x_test)
    data_y_disc = model_disc.predict(data_x_test)

        
    data_irt_items = [
        {"diff": diff, "disc": disc}
        for diff, disc in zip(data_y_diff, data_y_disc)
    ]


    items_joint = list(zip(data, data_irt_items))
    items_joint.sort(
        key=lambda x: fn_irt_utility(x[0], x[1], None, kwargs["fn_utility"]),
        reverse=True
    )

    items = [x[0] for x in items_joint]

    if return_model:
        return items, (model_diff, model_disc)
    else:
        return items

def cometsrc(data, model_path, reverse=False, **kwargs):
    import comet

    model = comet.load_from_checkpoint(model_path)
    scores = model.predict([
        {"src": line["src"]}
        for line in data
    ]).scores

    data_w_score = list(zip(data, scores))
    data_w_score.sort(key=lambda x: x[1], reverse=reverse)

    return [x[0] for x in data_w_score]

def cometsrc2(data, model_path1, model_path2, reverse=False, **kwargs):
    import comet

    model1 = comet.load_from_checkpoint(model_path1)
    scores1 = model1.predict([
        {"src": line["src"]}
        for line in data
    ]).scores
    model2 = comet.load_from_checkpoint(model_path2)
    scores2 = model2.predict([
        {"src": line["src"]}
        for line in data
    ]).scores
    scores = [s1*s2 for s1, s2 in zip(scores1, scores2)]

    data_w_score = list(zip(data, scores))
    data_w_score.sort(key=lambda x: x[1], reverse=reverse)

    return [x[0] for x in data_w_score]


def output_text_variance_unigram(data, **kwargs):
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
                out.append(2*(text_a & text_b).total()/(text_a.total()+text_b.total()))
        return np.average(out)

    # sort from smallest similarity
    data.sort(key=lambda line: _f(line), reverse=False)
    return data

def output_text_variance_bleu(data, **kwargs):
    import itertools
    import sacrebleu
    metric = sacrebleu.metrics.BLEU(effective_order=True)

    def _f(line):
        score = np.average([
            metric.sentence_score(
                text_a,
                [text_b],
            ).score
            for text_a, text_b in itertools.product(line["tgt"].values(), line["tgt"].values())
        ])
        return score

    # sort from smallest similarity
    data.sort(key=lambda line: _f(line), reverse=False)
    return data

def _run_simulation(args):
    data_old, data_prior = args
    data_new = random.sample(data_old, k=min(len(data_old), 20))
    clusters = utils.eval_system_clusters(data_new+data_prior, metric="MetricX-23-c")
    return data_new+data_prior, clusters

def synthetic_simulation(data, **kwargs):
    import multiprocessing

    data_new = []
    while len(data) > 0:
        print("Remaining", len(data))
        with multiprocessing.Pool(20) as pool:
            results = pool.map(_run_simulation, [[data, data_new]]*1000)

        # take best clustering but evaluate on human data
        data_new = max(results, key=lambda x: x[1])[0]
        data_best_i = {line["i"] for line in data_new}
        data = [line for line in data if line["i"] not in data_best_i]
    
    return data_new

METHODS = {
    "random": random_subset,
    "avg": metric_avg,
    "var": metric_var,
    "output_text_var": output_text_variance_bleu,
    "diversity": output_text_variance_bleu,
    "synthetic_simulation": synthetic_simulation,

    "pyirt_diff": partial(pyirt, fn_utility="diff"),
    "pyirt_disc": partial(pyirt, fn_utility="disc"),
    "pyirt_diffdisc": partial(pyirt, fn_utility="diffdisc"),
    "pyirt_feas": partial(pyirt, fn_utility="feas"),
    "pyirt_fic": partial(pyirt, fn_utility="fisher_information_content"),
    "pyirt_experiment": partial(pyirt, fn_utility="experiment"),

    "premlp_irt_diffdisc": partial(premlp_irt, fn_utility="diffdisc"),
    "premlp_irt_diff": partial(premlp_irt, fn_utility="diff"),
    "premlp_irt_disc": partial(premlp_irt, fn_utility="disc"),

    "premlp_var": partial(premlp_other, fn_utility=lambda line: np.var([sys_v["human"] for sys_v in line["scores"].values()])),
    "premlp_avg": partial(premlp_other, fn_utility=lambda line: np.average([sys_v["human"] for sys_v in line["scores"].values()])),

    # precomet_var val_pearson is semi-random, see comment in comet-src/experiments/03-generate_comet_data.py 
    "precomet_var": partial(cometsrc, model_path="/home/vilda/comet-src/lightning_logs/version_19777971/checkpoints/epoch=8-step=3519-val_pearson=0.009.ckpt", reverse=False),
    "precomet_avg": partial(cometsrc, model_path="/home/vilda/comet-src/lightning_logs/version_19777972/checkpoints/epoch=9-step=3910-val_pearson=0.150.ckpt", reverse=False),
    "precomet_div": partial(cometsrc, model_path="/home/vilda/comet-src/lightning_logs/version_19777784/checkpoints/epoch=5-step=2346-val_pearson=0.451.ckpt", reverse=False),
    "precomet_diff": partial(cometsrc, model_path="/cluster/work/sachan/vilem/comet-src/lightning_logs/version_18817024/checkpoints/epoch=1-step=1944-val_pearson=0.405.ckpt", reverse=False),
    "precomet_disc": partial(cometsrc, model_path="/cluster/work/sachan/vilem/comet-src/lightning_logs/version_18817064/checkpoints/epoch=1-step=1944-val_pearson=0.510.ckpt", reverse=False),
    "precomet_diffdisc": partial(
        cometsrc2,
        model_path1="/cluster/work/sachan/vilem/comet-src/lightning_logs/version_18817024/checkpoints/epoch=1-step=1944-val_pearson=0.405.ckpt",
        model_path2="/cluster/work/sachan/vilem/comet-src/lightning_logs/version_18817064/checkpoints/epoch=1-step=1944-val_pearson=0.510.ckpt",
        reverse=False
    ),
}