import os
from typing import Dict


PROPS = [x/100 for x in range(10, 90+1, 10)]

def load_data_squad(n_items=1000, n_systems=161):
    import json
    data = [json.loads(x) for x in open("data/squad_systems.jsonl")]
    data_out = []
    for pred_i, pred in enumerate(list(data[0]["predictions"].keys())[:n_items]):
        data_out.append({
            "i": pred_i,
            # TODO: check that all systems have the same predictions
            "scores": {
                sys["name"] + ":" + sys["submission_id"]: {
                    "exact_match": sys["predictions"][pred]["scores"]["exact_match"],
                    "f1": sys["predictions"][pred]["scores"]["f1"],
                }
                for sys in data[:n_systems]
            }
        })
    return data_out

def load_data_wmt(year="wmt23", langs="en-cs", normalize=False, binarize=False):
    import glob
    import collections
    line_src = open(f"data/mt-metrics-eval-v2/{year}/sources/{langs}.txt", "r").readlines()
    line_doc = open(f"data/mt-metrics-eval-v2/{year}/documents/{langs}.docs", "r").readlines()
    for fname in [
        f"data/mt-metrics-eval-v2/{year}/references/{langs}.refA.txt",
        f"data/mt-metrics-eval-v2/{year}/references/{langs}.refB.txt",
        f"data/mt-metrics-eval-v2/{year}/references/{langs}.refC.txt",
        f"data/mt-metrics-eval-v2/{year}/references/{langs}.refa.txt",
        f"data/mt-metrics-eval-v2/{year}/references/{langs}.refb.txt",
        f"data/mt-metrics-eval-v2/{year}/references/{langs}.refc.txt",
    ]:
        if os.path.exists(fname):
            line_ref = open(fname, "r").readlines()
            break
    line_sys = {}
    for f in glob.glob(f"data/mt-metrics-eval-v2/{year}/system-outputs/{langs}/*.txt"):
        sys = f.split("/")[-1].removesuffix(".txt")
        if sys in {"synthetic_ref", "refA", "chrf_bestmbr"}:
                continue

        line_sys[sys] = open(f, "r").readlines()

    systems = list(line_sys.keys())

    line_score = collections.defaultdict(list)
    for fname in [
        f"data/mt-metrics-eval-v2/{year}/human-scores/{langs}.da-sqm.seg.score",
        f"data/mt-metrics-eval-v2/{year}/human-scores/{langs}.wmt.seg.score",
        f"data/mt-metrics-eval-v2/{year}/human-scores/{langs}.mqm.seg.score",
        False
    ]:
        if fname and os.path.exists(fname):
            break
    
    if not fname:
        # did not find human scores
        return []
    
    for line_raw in open(fname, "r").readlines():
        sys, score = line_raw.strip().split()
        line_score[sys].append({"human": score})


    for f in glob.glob(f"data/mt-metrics-eval-v2/{year}/metric-scores/{langs}/*-refA.seg.score"):
        metric = f.split("/")[-1].removesuffix("-refA.seg.score")
        for line_i, line_raw in enumerate(open(f, "r").readlines()):
            sys, score = line_raw.strip().split("\t")
            # for refA, refB, synthetic_ref, and other "systems" not evaluated
            # TODO: maybe remove those from the systems list?
            if sys not in line_score:
                continue
            # NOTE: there's no guarantee that this indexing is correct
            line_score[sys][line_i % len(line_src)][metric] = float(score)

    # filter out lines that have no human score
    line_score = {k:v for k,v in line_score.items() if len(v) > 0}
    systems = [sys for sys in systems if sys in line_score]

    # putting it all together
    data = []
    line_id_true = 0
    for line_i, (line_src, line_ref, line_doc) in enumerate(zip(line_src, line_ref, line_doc)):
        # filter None on the whole row
        if any([line_score[sys][line_i]["human"] in {"None", "0"} for sys in systems]):
        # if any([line_score[sys][line_i]["human"] in {"None"} for sys in systems]):
            continue

        line_domain, line_doc = line_doc.strip().split("\t")

        data.append({
            "i": line_id_true,
            "src": line_src.strip(),
            "ref": line_ref.strip(),
            "tgt": {sys: line_sys[sys][line_i].strip() for sys in systems},
            "domain": line_domain,
            "scores": {sys: {metric: float(v) for metric,v in line_score[sys][line_i].items()} for sys in systems},
        })
        line_id_true += 1
    

    if normalize and not binarize:
        # if we are binarizing, none of this matters
        import collections
        data_flat = collections.defaultdict(list)
        for line in data:
            for met_all in line["scores"].values():
                for met_k, met_v in met_all.items():
                    data_flat[met_k].append(met_v)

        # normalize
        data_flat = {
            k: (min(v), max(v))
            for k,v in data_flat.items()
        }

        for line in data:
            for sys, met_all in line["scores"].items():
                for met_k, met_v in met_all.items():
                    # (x-min)/(max-min) normalize
                    line["scores"][sys][met_k] = (met_v-data_flat[met_k][0])/(data_flat[met_k][1]-data_flat[met_k][0])

    if binarize:
        import collections
        import numpy as np
        data_flat = collections.defaultdict(list)
        for line in data:
            for met_all in line["scores"].values():
                for met_k, met_v in met_all.items():
                    data_flat[met_k].append(met_v)

        # normalize
        data_flat = {
            k: np.median(v)
            for k,v in data_flat.items()
        }

        for line in data:
            for sys, met_all in line["scores"].items():
                for met_k, met_v in met_all.items():
                    line["scores"][sys][met_k] = 1*(met_v >= data_flat[met_k])
    
    # import sys
    # print("Loaded", len(data), "lines of", len(systems), "systems", file=sys.stderr)
    return data

def load_data_wmt_all(**kwargs):
    data = {
        args: load_data_wmt(*args, **kwargs)
        for args in [
            ("wmt23.sent", "en-de"),
            ("wmt23", "cs-uk"),
            ("wmt23", "de-en"),
            ("wmt23", "en-cs"),
            ("wmt23", "en-de"),
            ("wmt23", "en-he"),
            ("wmt23", "en-ja"),
            ("wmt23", "en-ru"),
            ("wmt23", "en-uk"),
            ("wmt23", "en-zh"),
            ("wmt23", "he-en"),
            ("wmt23", "ja-en"),
            ("wmt23", "ru-en"),
            ("wmt23", "uk-en"),
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
    return {k:v for k,v in data.items() if len(v) > 0}

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

def eval_system_clusters(data: list, metric="human"):
    from scipy.stats import wilcoxon
    # computes number of clusters

    # sort from top
    sys_ord = list(get_sys_absolute(data).items())
    sys_ord.sort(key=lambda x: x[1], reverse=True)
    sys_ord = [sys for sys, _ in sys_ord]

    def get_scores(system):
        return [line["scores"][system][metric] for line in data]

    clusters = [[get_scores(sys_ord.pop(0))]]
    while sys_ord:
        sys_scores = get_scores(sys_ord.pop(0))
        # TODO: should this be clusters[-1][0] or clusters[-1][-1]?
        diffs = [x - y for x, y in zip(sys_scores, clusters[-1][0])]
        if wilcoxon(diffs, alternative="less").pvalue < 0.05:
            clusters.append([sys_scores])
        else:
            clusters[-1].append(sys_scores)
    return len(clusters)

def eval_order_accuracy(data_new: list, data_old: list, metric="human"):
    # evaluates against ordering from data_old
    import itertools
    import numpy as np
    
    systems = list(data_old[0]["scores"].keys())

    scores_old = get_sys_absolute(data_old, metric=metric)
    scores_new = get_sys_absolute(data_new, metric=metric)

    result = []
    for sys1, sys2 in itertools.combinations(systems, 2):
        result.append((scores_old[sys1]<scores_old[sys2])==(scores_new[sys1]<scores_new[sys2]))

    return np.average(result)

def get_ord_accuracy(ord1, ord2):
    import itertools
    import numpy as np

    systems = list(ord1.keys())
    result = []

    for sys1, sys2 in itertools.combinations(systems, 2):
        result.append((ord2[sys1]<ord2[sys2])==(ord1[sys1]<ord1[sys2]))

    return np.average(result)

def get_nice_subset(data_old, target_size=100, step_size=10, metric="human"):
    import numpy as np
    order_full = get_sys_ordering(data_old, metric=metric)

    print(f"Previous average accuracy: {np.average([get_ord_accuracy(order_full, get_sys_ordering([line], metric=metric)) for line in data_old]):.2%}")

    while len(data_old) > target_size:
        order_full = get_sys_ordering(data_old, metric=metric)
        data_old.sort(key=lambda line: get_ord_accuracy(order_full, get_sys_ordering([line], metric=metric)))
        data_old = data_old[step_size:]

    print(f"New average accuracy: {np.average([get_ord_accuracy(order_full, get_sys_ordering([line], metric=metric)) for line in data_old]):.2%}")
    return data_old

def pred_irt(system_theta, item):
    import numpy as np
    if "feas" in item:
        return item["feas"] + (1 - item["feas"]) / (1 + np.exp(-item["disc"] * (system_theta - item["diff"])))
    if "disc" in item:
        return 1 / (1 + np.exp(-item["disc"] * (system_theta - item["diff"])))
    if "diff" in item:
        return 1 / (1 + np.exp(system_theta - item["diff"]))
    raise Exception("Uknown item", item)