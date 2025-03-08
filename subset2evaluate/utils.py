import importlib.metadata
from typing import List, Optional, Union
from typing import Dict
import numpy as np
import subset2evaluate
PROPS = np.geomspace(0.05, 0.5, 10)


def _data_minmax_normalize(data):
    """
    In-place min-max normalization of all scores
    """
    import collections
    # if we are binarizing, none of this matters
    data_flat = collections.defaultdict(list)
    for line in data:
        for met_all in line["scores"].values():
            for met_k, met_v in met_all.items():
                data_flat[met_k].append(met_v)

    # normalize
    data_flat = {
        k: (min(v), max(v))
        for k, v in data_flat.items()
    }

    for line in data:
        for model, met_all in line["scores"].items():
            for met_k, met_v in met_all.items():
                # (x-min)/(max-min) normalize
                line["scores"][model][met_k] = (met_v - data_flat[met_k][0]) / (data_flat[met_k][1] - data_flat[met_k][0])


def _data_median_binarize(data):
    """
    In-place median binarization of all scores
    """
    import collections
    data_flat = collections.defaultdict(list)
    for line in data:
        for met_all in line["scores"].values():
            for met_k, met_v in met_all.items():
                data_flat[met_k].append(met_v)

    # normalize
    data_flat = {
        k: np.median(v)
        for k, v in data_flat.items()
    }

    for line in data:
        for model, met_all in line["scores"].items():
            for met_k, met_v in met_all.items():
                line["scores"][model][met_k] = 1 * (met_v >= data_flat[met_k])


def ensure_wmt_exists():
    import requests
    import os
    import tarfile

    if not os.path.exists("data/mt-metrics-eval-v2/"):
        print("Downloading WMT data because data/mt-metrics-eval-v2/ does not exist..")
        os.makedirs("data/", exist_ok=True)
        r = requests.get("https://storage.googleapis.com/mt-metrics-eval/mt-metrics-eval-v2.tgz")
        with open("data/mt-metrics-eval-v2.tgz", "wb") as f:
            f.write(r.content)
        with tarfile.open("data/mt-metrics-eval-v2.tgz", "r:gz") as f:
            f.extractall("data/")
        os.remove("data/mt-metrics-eval-v2.tgz")


def load_data_wmt(  # noqa: C901
    year: str = "wmt23",
    langs: str = "en-cs",
    normalize: bool = True,
    binarize: bool = False,
    file_protocol: Optional[str] = None,
    file_reference: Optional[str] = None,
    zero_bad: bool = False,
):
    import glob
    import collections
    import numpy as np
    import os
    import pickle
    import contextlib
    import importlib.metadata

    # temporarily change to the root directory, this requires Python 3.11
    with contextlib.chdir(os.path.dirname(os.path.realpath(__file__)) + "/../"):
        ensure_wmt_exists()

        os.makedirs("data/cache/", exist_ok=True)
        cache_f = f"data/cache/{year}_{langs}_n{int(normalize)}_b{int(binarize)}_zb{int(zero_bad)}_fp{file_protocol}_fr{file_reference}.pkl"

        # load cache if exists
        if os.path.exists(cache_f):
            with open(cache_f, "rb") as f:
                cache = pickle.load(f)
                # only load data if they come from the same version
                if isinstance(cache, dict) and "version" in cache.keys() and cache["version"] == importlib.metadata.version("subset2evaluate"):
                    return cache["data"]
        

        lines_src = open(f"data/mt-metrics-eval-v2/{year}/sources/{langs}.txt", "r").readlines()
        lines_doc = open(f"data/mt-metrics-eval-v2/{year}/documents/{langs}.docs", "r").readlines()
        lines_ref = None

        if file_reference is not None:
            f_references = [
                f"data/mt-metrics-eval-v2/{year}/references/{langs}.{file_reference}.txt",
            ]
        else:
            f_references = [
                f"data/mt-metrics-eval-v2/{year}/references/{langs}.refA.txt",
                f"data/mt-metrics-eval-v2/{year}/references/{langs}.refB.txt",
                f"data/mt-metrics-eval-v2/{year}/references/{langs}.refC.txt",
                f"data/mt-metrics-eval-v2/{year}/references/{langs}.refa.txt",
                f"data/mt-metrics-eval-v2/{year}/references/{langs}.refb.txt",
                f"data/mt-metrics-eval-v2/{year}/references/{langs}.refc.txt",
                f"data/mt-metrics-eval-v2/{year}/references/{langs}.ref.txt",
            ]

        for fname in [*f_references, False]:
            if os.path.exists(fname):
                break
        if not fname:
            # did not find reference
            return []

        lines_ref = open(fname, "r").readlines()

        # do not consider canary line
        contain_canary_line = lines_src[0].lower().startswith("canary")
        if contain_canary_line:
            lines_src.pop(0)
            lines_doc.pop(0)
            lines_ref.pop(0)


        line_model = {}
        for f in glob.glob(f"data/mt-metrics-eval-v2/{year}/system-outputs/{langs}/*.txt"):
            model = f.split("/")[-1].removesuffix(".txt")
            if model in {"synthetic_ref", "refA", "chrf_bestmbr"}:
                continue

            line_model[model] = open(f, "r").readlines()
            if contain_canary_line:
                line_model[model].pop(0)

        models = list(line_model.keys())

        lines_score = collections.defaultdict(list)

        if file_protocol is not None:
            f_protocols = [
                f"data/mt-metrics-eval-v2/{year}/human-scores/{langs}.{file_protocol}.seg.score",
            ]
        else:
            f_protocols = [
                f"data/mt-metrics-eval-v2/{year}/human-scores/{langs}.esa.seg.score",
                f"data/mt-metrics-eval-v2/{year}/human-scores/{langs}.da-sqm.seg.score",
                f"data/mt-metrics-eval-v2/{year}/human-scores/{langs}.mqm.seg.score",
                f"data/mt-metrics-eval-v2/{year}/human-scores/{langs}.wmt.seg.score",
                f"data/mt-metrics-eval-v2/{year}/human-scores/{langs}.appraise.seg.score",
                f"data/mt-metrics-eval-v2/{year}/human-scores/{langs}.wmt-raw.seg.score",
                f"data/mt-metrics-eval-v2/{year}/human-scores/{langs}.wmt-appraise.seg.score",
            ]
        for fname in [*f_protocols, False]:
            if fname and os.path.exists(fname):
                break

        if not fname:
            # did not find human scores
            return []

        for line_raw in open(fname, "r").readlines():
            model, score = line_raw.strip().split()
            lines_score[model].append({"human": score})
        if contain_canary_line:
            for model in lines_score:
                lines_score[model].pop(0)

        total_n_srcs = len(lines_src)
        if contain_canary_line:
            total_n_srcs += 1
        for f in glob.glob(f"data/mt-metrics-eval-v2/{year}/metric-scores/{langs}/*-refA.seg.score"):
            metric = f.split("/")[-1].removesuffix("-refA.seg.score")
            for line_i, line_raw in enumerate(open(f, "r").readlines()):
                model, score = line_raw.strip().split("\t")
                # for refA, refB, synthetic_ref, and other "modeltems" not evaluated
                # NOTE: another option is remove the *models*
                if model not in lines_score:
                    continue

                model_line_i = line_i % total_n_srcs
                if contain_canary_line:
                    # do not include canary line scores
                    if model_line_i == 0:
                        continue

                    model_line_i -= 1

                lines_score[model][model_line_i][metric] = float(score)

        # filter out lines that have no human score
        lines_score = {k: v for k, v in lines_score.items() if len(v) > 0}
        models = [model for model in models if model in lines_score]

        # putting it all together
        data = []
        line_id_true = 0

        # remove models that have no outputs
        ANNOTATION_BAD = {"None"}
        if zero_bad:
            ANNOTATION_BAD |= {"0"}
        models_bad = set()
        for model, scores in lines_score.items():
            if all([x["human"] in ANNOTATION_BAD for x in scores]):
                models_bad.add(model)
        models = [model for model in models if model not in models_bad]

        for line_i, (line_src, line_ref, line_doc) in enumerate(zip(lines_src, lines_ref, lines_doc)):
            # filter None on the whole row
            # TODO: maybe still consider segments with 0?
            # NOTE: if we do that, then we won't have metrics annotations for all segments, which is bad
            if any([lines_score[model][line_i]["human"] in ANNOTATION_BAD for model in models]):
                continue
            # metrics = set(lines_score[models[0]][line_i].keys())
            # # if we're missing some metric, skip the line
            # if any([set(lines_score[model][line_i].keys()) != metrics for model in models]):
            #     continue

            line_domain, line_doc = line_doc.strip().split("\t")
            # for CJK languages, we need to count characters
            if "ja" in langs or "zh" in langs or "ko" in langs or "th" in langs:
                word_count = len(line_src.strip())
            else:
                word_count = len(line_src.strip().split())

            data.append({
                "i": line_id_true,
                "src": line_src.strip(),
                "ref": line_ref.strip(),
                "tgt": {model: line_model[model][line_i].strip() for model in models},
                # just very rough estimate, the coefficients don't matter because it'll be normalized later anyway
                "cost": 0.15 * word_count + 33.7,
                "domain": line_domain,
                "doc": line_doc,
                "scores": {model: {metric: float(v) for metric, v in lines_score[model][line_i].items()} for model in models},
            })
            line_id_true += 1

        # normalize times
        if data:
            data_flat = [line["cost"] for line in data]
            cost_avg = np.average(data_flat)
            cost_std = np.std(data_flat)
            for line in data:
                # z-normalize and make mean 1
                line["cost"] = (line["cost"] - cost_avg) / cost_std + 1

            data_flat = [line["cost"] for line in data]
            cost_min = np.min(data_flat)
            for line in data:
                # make sure it's positive
                line["cost"] = (line["cost"] - cost_min) / (1 - cost_min)

        # this is min-max normalization
        if normalize and not binarize:
            _data_minmax_normalize(data)

        if binarize:
            _data_median_binarize(data)

        # save cache
        with open(cache_f, "wb") as f:
            pickle.dump({"version": importlib.metadata.version("subset2evaluate"), "data": data}, f)

    return data


def load_data_wmt_all(min_items=500, **kwargs):
    data = {
        args: load_data_wmt(*args, **kwargs)
        for args in [
            ("wmt23.sent", "en-de"),
            ("wmt23", "cs-uk"),
            ("wmt23", "de-en"),
            ("wmt23", "en-cs"),
            ("wmt23", "en-de"),
            ("wmt23", "en-ja"),
            ("wmt23", "en-zh"),
            ("wmt23", "he-en"),
            ("wmt23", "ja-en"),
            ("wmt23", "zh-en"),

            # NOTE: intentionally not the first so that [:9] is reserved for evaluation
            ("wmt24", "cs-uk"),
            ("wmt24", "en-cs"),
            ("wmt24", "en-de"),
            ("wmt24", "en-es"),
            ("wmt24", "en-hi"),
            ("wmt24", "en-is"),
            ("wmt24", "en-ja"),
            ("wmt24", "en-ru"),
            ("wmt24", "en-uk"),
            ("wmt24", "en-zh"),
            ("wmt24", "ja-zh"),

            ("wmt22", "cs-en"),
            ("wmt22", "cs-uk"),
            ("wmt22", "de-en"),
            ("wmt22", "en-cs"),
            ("wmt22", "en-de"),
            ("wmt22", "en-hr"),
            ("wmt22", "en-ja"),
            ("wmt22", "en-liv"),
            ("wmt22", "en-ru"),
            ("wmt22", "en-uk"),
            ("wmt22", "en-zh"),
            ("wmt22", "ja-en"),
            ("wmt22", "liv-en"),
            ("wmt22", "ru-en"),
            ("wmt22", "sah-ru"),
            ("wmt22", "uk-cs"),
            ("wmt22", "uk-en"),
            ("wmt22", "zh-en"),

            ("wmt21.tedtalks", "en-de"),
            ("wmt21.tedtalks", "en-ru"),
            ("wmt21.tedtalks", "zh-en"),
            ("wmt21.news", "cs-en"),
            ("wmt21.news", "de-en"),
            ("wmt21.news", "de-fr"),
            ("wmt21.news", "en-cs"),
            ("wmt21.news", "en-de"),
            ("wmt21.news", "en-ha"),
            ("wmt21.news", "en-is"),
            ("wmt21.news", "en-ja"),
            ("wmt21.news", "en-ru"),
            ("wmt21.news", "en-zh"),
            ("wmt21.news", "fr-de"),
            ("wmt21.news", "ha-en"),
            ("wmt21.news", "is-en"),
            ("wmt21.news", "ja-en"),
            ("wmt21.news", "ru-en"),
            ("wmt21.news", "zh-en"),
            ("wmt21.flores", "bn-hi"),
            ("wmt21.flores", "hi-bn"),
            ("wmt21.flores", "xh-zu"),
            ("wmt21.flores", "zu-xh"),

            ("wmt20", "km-en"),
            ("wmt20", "en-zh"),
            ("wmt20", "ps-en"),
            ("wmt20", "zh-en"),
            ("wmt20", "ru-en"),
            ("wmt20", "iu-en"),
            ("wmt20", "ta-en"),
            ("wmt20", "en-ta"),
            ("wmt20", "en-cs"),
            ("wmt20", "de-en"),
            ("wmt20", "en-de"),
            ("wmt20", "en-ja"),
            ("wmt20", "cs-en"),
            ("wmt20", "en-pl"),
            ("wmt20", "pl-en"),
            ("wmt20", "en-ru"),
            ("wmt20", "en-iu"),
            ("wmt20", "ja-en"),

            ("wmt19", "en-zh"),
            ("wmt19", "en-lt"),
            ("wmt19", "en-gu"),
            ("wmt19", "ru-en"),
            ("wmt19", "kk-en"),
            ("wmt19", "en-fi"),
            ("wmt19", "zh-en"),
            ("wmt19", "fi-en"),
            ("wmt19", "en-cs"),
            ("wmt19", "de-en"),
            ("wmt19", "en-ru"),
            ("wmt19", "gu-en"),
            ("wmt19", "en-kk"),
            ("wmt19", "lt-en"),
            ("wmt19", "de-fr"),
            ("wmt19", "fr-de"),
            ("wmt19", "en-de"),
            ("wmt19", "de-cs"),
        ]
    }
    # filter out empty datasets
    # some years/langs have issues with human annotations coverage
    return {k: v for k, v in data.items() if len(v) > min_items}


def load_data_summeval(  # noqa: C901
    normalize: str = True,
    load_extra: str = False,
):
    from datasets import load_dataset
    from functools import reduce
    import collections
    import contextlib
    import os
    import pickle

    # temporarily change to the root directory, this requires Python 3.11
    with contextlib.chdir(os.path.dirname(os.path.realpath(__file__)) + "/../"):
        os.makedirs("data/cache/", exist_ok=True)
        cache_f = f"data/cache/summeval_n{int(normalize)}_l{int(load_extra)}.pkl"

        # load cache if exists
        if os.path.exists(cache_f):
            with open(cache_f, "rb") as f:
                return pickle.load(f)

    data_raw = load_dataset("KnutJaegersberg/summeval_pairs")["train"]

    data_by_id = collections.defaultdict(list)
    for line in data_raw:
        data_by_id[line["id"]].append(line)

    def avg_human_annotations(expert_annotations: List[Dict[str, float]]) -> Dict[str, float]:
        scores = collections.defaultdict(list)
        for line in expert_annotations:
            for k, v in line.items():
                scores[k].append(v)
        scores = {"human_" + k: sum(v) / len(v) for k, v in scores.items()}
        scores_values = list(scores.values())

        # multiply all human
        scores["human_sum"] = reduce(lambda x, y: x + y, scores_values)
        scores["human_mul"] = reduce(lambda x, y: x * y, scores_values)
        return scores

    data = []
    for i, v in data_by_id.items():
        data.append({
            "i": i,
            "src": v[0]["text"],
            "ref": None,
            "tgt": {line["model_id"]: line["decoded"] for line in v},
            "scores": {
                # rouge is nested for some reason
                line["model_id"]: (
                    line["metric_scores_1"] | line["metric_scores_1"]["rouge"] | avg_human_annotations(line["expert_annotations"])
                )
                for line in v
            },
        })

    # remove rouge from scores and fix supert
    data = [
        {
            **line,
            "scores": {
                model: {
                    metric:
                    score if metric != "supert" else score[0]
                    for metric, score in metrics.items()
                    if metric != "rouge"
                }
                for model, metrics in line["scores"].items()
            }
        }
        for line in data
    ]

    if load_extra:
        # temporarily change to the root directory, this requires Python 3.11
        with contextlib.chdir(os.path.dirname(os.path.realpath(__file__)) + "/../"):
            # TODO: in the future these files need to be stored somewhere statically
            data_metrics = load_data("../subset2evaluate-tmp/data_other/summeval_gpt.jsonl")

        data_metrics_i = {
            x["i"]: x
            for x in data_metrics
        }
        assert all(x["i"] in data_metrics_i for x in data)
        for x in data:
            x["scores"] = {
                sys: data_metrics_i[x["i"]]["scores"][sys] | v
                for sys, v in x["scores"].items()
                if sys in data_metrics_i[x["i"]]["scores"]
            }
            x["scores"] = {
                sys: v | {
                    "gpt_sum": v["gpt_relevance"] + v["gpt_coherence"] + v["gpt_consistency"] + v["gpt_fluency"],
                    "gpt_mul": v["gpt_relevance"] * v["gpt_coherence"] * v["gpt_consistency"] * v["gpt_fluency"],
                }
                for sys, v in x["scores"].items()
            }

        # temporarily change to the root directory, this requires Python 3.11
        with contextlib.chdir(os.path.dirname(os.path.realpath(__file__)) + "/../"):
            # TODO: in the future these files need to be stored somewhere statically
            data_metrics = load_data("../subset2evaluate-tmp/data_other/summeval_unieval.jsonl")

        data_metrics_i = {
            x["i"]: x
            for x in data_metrics
        }
        assert all(x["i"] in data_metrics_i for x in data)
        for x in data:
            x["scores"] = {
                sys: data_metrics_i[x["i"]]["scores"][sys] | v
                for sys, v in x["scores"].items()
                if sys in data_metrics_i[x["i"]]["scores"]
            }
            x["scores"] = {
                sys: v | {
                    "unieval_sum": (
                        v["unieval_relevance"] + v["unieval_coherence"] + v["unieval_consistency"] + v["unieval_fluency"]
                    ),
                    "unieval_mul": (
                        v["unieval_relevance"] * v["unieval_coherence"] * v["unieval_consistency"] * v["unieval_fluency"]
                    ),
                }
                for sys, v in x["scores"].items()
            }

    if normalize:
        _data_minmax_normalize(data)

    # temporarily change to the root directory, this requires Python 3.11
    with contextlib.chdir(os.path.dirname(os.path.realpath(__file__)) + "/../"):
        # save cache
        with open(cache_f, "wb") as f:
            pickle.dump(data, f)

    return data


def load_data_rose():
    import datasets
    data_raw = datasets.load_dataset("Salesforce/rose", "cnndm_protocol")
    data_raw = list(data_raw.values())[0]
    data = []
    for line in data_raw:
        data.append({
            "i": line["example_id"],
            "src": line["source"],
            "ref": line["reference"],
            "tgt": line["model_outputs"],
            # NOTE: no metrics!
            "scores": line["annotations"],
        })


def pred_irt(model_theta, item):
    import numpy as np
    if "feas" in item:
        # NOTE: true for 4PL, not for 3PL
        # return  item["feas"] / (1 + np.exp(-item["disc"] * (model_theta - item["diff"])))
        return item["feas"] + (1 - item["feas"]) / (1 + np.exp(-item["disc"] * (model_theta - item["diff"])))
    if "disc" in item:
        return 1 / (1 + np.exp(-item["disc"] * (model_theta - item["diff"])))
    if "diff" in item:
        return 1 / (1 + np.exp(model_theta - item["diff"]))
    raise Exception("Uknown item", item)


def sanitize_data(data: List[Dict], top_systems=5):
    """
    Makes sure that all items contain the same systems.
    """
    import collections

    system_counter = collections.Counter()
    for line in data:
        for system in line["scores"].keys():
            system_counter[system] += 1

    systems = {x[0] for x in system_counter.most_common(top_systems)}

    # filter items that don't have these systems
    data = [
        line for line in data
        if (
            all(system in line["scores"] for system in systems) and
            all(system in line["tgt"] for system in systems)
        )
    ]

    # filter systems that are not everywhere
    systems = set(data[0]["scores"].keys())
    for line in data:
        systems = systems.intersection(set(line["scores"].keys()))

    # filter other systems
    data = [
        {
            **line,
            "scores": {
                system: metrics
                for system, metrics in line["scores"].items()
                if system in systems
            },
            "tgt": {
                system: tgt
                for system, tgt in line["tgt"].items()
                if system in systems
            }
        }
        for line in data
    ]
    return data


def load_data(data: Union[List, str], **kwargs):
    import os
    import json

    if type(data) is list:
        pass
    elif os.path.exists(data):
        return [json.loads(x) for x in open(data, "r")]
    elif data.startswith("wmt"):
        data_year, data_lang = data.split("/")
        if data_year == "wmt" and data_lang == "all":
            data = load_data_wmt_all(**kwargs)
        else:
            data = load_data_wmt(year=data_year, langs=data_lang, **kwargs)
    elif data == "summeval":
        return load_data_summeval(**kwargs)
    else:
        raise Exception("Could not parse data")

    return data
