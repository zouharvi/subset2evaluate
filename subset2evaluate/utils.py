from typing import List, Union
from typing import Dict
import numpy as np
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
        for sys, met_all in line["scores"].items():
            for met_k, met_v in met_all.items():
                # (x-min)/(max-min) normalize
                line["scores"][sys][met_k] = (met_v - data_flat[met_k][0]) / (data_flat[met_k][1] - data_flat[met_k][0])


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
        for sys, met_all in line["scores"].items():
            for met_k, met_v in met_all.items():
                line["scores"][sys][met_k] = 1 * (met_v >= data_flat[met_k])


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


def load_data_wmt(year="wmt23", langs="en-cs", normalize=True, binarize=False):
    import glob
    import collections
    import numpy as np
    import os
    import pickle
    import contextlib

    # temporarily change to the root directory
    with contextlib.chdir(os.path.dirname(os.path.realpath(__file__)) + "/../"):
        ensure_wmt_exists()

        os.makedirs("data/cache/", exist_ok=True)
        cache_f = f"data/cache/{year}_{langs}_n{int(normalize)}_b{int(binarize)}.pkl"

        # load cache if exists
        if os.path.exists(cache_f):
            with open(cache_f, "rb") as f:
                return pickle.load(f)

        lines_src = open(f"data/mt-metrics-eval-v2/{year}/sources/{langs}.txt", "r").readlines()
        lines_doc = open(f"data/mt-metrics-eval-v2/{year}/documents/{langs}.docs", "r").readlines()
        lines_ref = None
        for fname in [
            f"data/mt-metrics-eval-v2/{year}/references/{langs}.refA.txt",
            f"data/mt-metrics-eval-v2/{year}/references/{langs}.refB.txt",
            f"data/mt-metrics-eval-v2/{year}/references/{langs}.refC.txt",
            f"data/mt-metrics-eval-v2/{year}/references/{langs}.refa.txt",
            f"data/mt-metrics-eval-v2/{year}/references/{langs}.refb.txt",
            f"data/mt-metrics-eval-v2/{year}/references/{langs}.refc.txt",
            f"data/mt-metrics-eval-v2/{year}/references/{langs}.ref.txt",
        ]:
            if os.path.exists(fname):
                lines_ref = open(fname, "r").readlines()
                break
        if lines_ref is None:
            return []

        line_sys = {}
        for f in glob.glob(f"data/mt-metrics-eval-v2/{year}/system-outputs/{langs}/*.txt"):
            sys = f.split("/")[-1].removesuffix(".txt")
            if sys in {"synthetic_ref", "refA", "chrf_bestmbr"}:
                continue

            line_sys[sys] = open(f, "r").readlines()

        systems = list(line_sys.keys())

        lines_score = collections.defaultdict(list)
        for fname in [
            f"data/mt-metrics-eval-v2/{year}/human-scores/{langs}.da-sqm.seg.score",
            f"data/mt-metrics-eval-v2/{year}/human-scores/{langs}.mqm.seg.score",
            f"data/mt-metrics-eval-v2/{year}/human-scores/{langs}.wmt.seg.score",
            f"data/mt-metrics-eval-v2/{year}/human-scores/{langs}.appraise.seg.score",
            f"data/mt-metrics-eval-v2/{year}/human-scores/{langs}.wmt-raw.seg.score",
            f"data/mt-metrics-eval-v2/{year}/human-scores/{langs}.wmt-appraise.seg.score",
            False
        ]:
            if fname and os.path.exists(fname):
                break

        if not fname:
            # did not find human scores
            return []

        for line_raw in open(fname, "r").readlines():
            sys, score = line_raw.strip().split()
            lines_score[sys].append({"human": score})

        for f in glob.glob(f"data/mt-metrics-eval-v2/{year}/metric-scores/{langs}/*-refA.seg.score"):
            metric = f.split("/")[-1].removesuffix("-refA.seg.score")
            for line_i, line_raw in enumerate(open(f, "r").readlines()):
                sys, score = line_raw.strip().split("\t")
                # for refA, refB, synthetic_ref, and other "systems" not evaluated
                # TODO: maybe remove those from the systems list?
                if sys not in lines_score:
                    continue
                # NOTE: there's no guarantee that this indexing is correct
                lines_score[sys][line_i % len(lines_src)][metric] = float(score)

        # filter out lines that have no human score
        lines_score = {k: v for k, v in lines_score.items() if len(v) > 0}
        systems = [sys for sys in systems if sys in lines_score]

        # putting it all together
        data = []
        line_id_true = 0

        for line_i, (line_src, line_ref, line_doc) in enumerate(zip(lines_src, lines_ref, lines_doc)):
            # filter None on the whole row
            # TODO: maybe still consider segments with 0?
            # NOTE: if we do that, then we won't have metrics annotations for all segments, which is bad
            if any([lines_score[sys][line_i]["human"] in {"None", "0"} for sys in systems]):
                continue
            # metrics = set(lines_score[systems[0]][line_i].keys())
            # # if we're missing some metric, skip the line
            # if any([set(lines_score[sys][line_i].keys()) != metrics for sys in systems]):
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
                "tgt": {sys: line_sys[sys][line_i].strip() for sys in systems},
                # just very rough estimate, the coefficients don't matter because it'll be normalized later anyway
                "time": 0.15 * word_count + 33.7,
                "domain": line_domain,
                "doc": line_doc,
                "scores": {sys: {metric: float(v) for metric, v in lines_score[sys][line_i].items()} for sys in systems},
            })
            line_id_true += 1

        # normalize times
        if data:
            data_flat = [line["time"] for line in data]
            mean = np.average(data_flat)
            std = np.std(data_flat)
            for line in data:
                # make it have var=1 and avg=0
                # line["time"] = (line["time"]-data_flat[0])/(data_flat[1]-data_flat[0]) + 1
                # z-normalize
                line["time"] = (line["time"] - mean) / std + 1

        # this is min-max normalization
        if normalize and not binarize:
            _data_minmax_normalize(data)

        if binarize:
            _data_median_binarize(data)

        # save cache
        with open(cache_f, "wb") as f:
            pickle.dump(data, f)

    return data


def load_data_wmt_all(min_segments=500, **kwargs):
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
    return {k: v for k, v in data.items() if len(v) > min_segments}


def load_data_summeval(normalize=True):
    from datasets import load_dataset
    from functools import reduce
    import collections
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

        # multiply all human
        scores["human_all"] = reduce(lambda x, y: x * y, scores.values())
        return scores

    data = []
    for i, v in data_by_id.items():
        # "coherence": 2, "consistency": 1, "fluency": 4, "relevance": 2
        data.append({
            "i": i,
            "src": None,
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
                sys: {
                    metric:
                    score if metric != "supert" else score[0]
                    for metric, score in metrics.items()
                    if metric != "rouge"
                }
                for sys, metrics in line["scores"].items()
            }
        }
        for line in data
    ]

    if normalize:
        _data_minmax_normalize(data)

    return data


def pred_irt(system_theta, item):
    import numpy as np
    if "feas" in item:
        # NOTE: true for 4PL, not for 3PL
        # return  item["feas"] / (1 + np.exp(-item["disc"] * (system_theta - item["diff"])))
        return item["feas"] + (1 - item["feas"]) / (1 + np.exp(-item["disc"] * (system_theta - item["diff"])))
    if "disc" in item:
        return 1 / (1 + np.exp(-item["disc"] * (system_theta - item["diff"])))
    if "diff" in item:
        return 1 / (1 + np.exp(system_theta - item["diff"]))
    raise Exception("Uknown item", item)


def load_data(data: Union[List, str]):
    import os
    import json

    if type(data) is list:
        pass
    elif os.path.exists(data):
        return [json.loads(x) for x in open(data, "r")]
    elif data.startswith("wmt"):
        data_year, data_lang = data.split("/")
        if data_lang == "all":
            data = load_data_wmt_all()
        else:
            data = load_data_wmt(year=data_year, langs=data_lang, normalize=True)
    elif data == "summeval":
        return load_data_summeval(normalize=True)
    else:
        raise Exception("Could not parse data")

    return data
