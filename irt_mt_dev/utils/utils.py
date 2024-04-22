from typing import Dict


PROPS = [x/100 for x in range(10, 90+1, 10)]

def load_data(year="wmt23", langs="en-cs", normalize=False, systems=None):
    import glob
    import collections
    line_src = open(f"data/mt-metrics-eval-v2/{year}/sources/{langs}.txt", "r").readlines()
    line_doc = open(f"data/mt-metrics-eval-v2/{year}/documents/{langs}.docs", "r").readlines()
    line_ref = open(f"data/mt-metrics-eval-v2/{year}/references/{langs}.refA.txt", "r").readlines()
    line_sys = {}
    for f in glob.glob(f"data/mt-metrics-eval-v2/{year}/system-outputs/{langs}/*.txt"):
        sys = f.split("/")[-1].removesuffix(".txt")
        if sys in {"synthetic_ref", "refA"}:
                continue

        line_sys[sys] = open(f, "r").readlines()

    if systems is None:
        systems = list(line_sys.keys())
    else:
        assert type(systems) == list

    line_score = collections.defaultdict(list)
    for line_raw in open(f"data/mt-metrics-eval-v2/{year}/human-scores/{langs}.da-sqm.seg.score", "r").readlines():
        sys, score = line_raw.strip().split("\t")
        line_score[sys].append({"score": score})

    for f in glob.glob(f"data/mt-metrics-eval-v2/{year}/metric-scores/{langs}/*-refA.seg.score"):
        metric = f.split("/")[-1].removesuffix("-refA.seg.score")
        for line_i, line_raw in enumerate(open(f, "r").readlines()):
            sys, score = line_raw.strip().split("\t")
            if sys in {"synthetic_ref", "refA"}:
                continue
            # NOTE: there's no guarantee that this indexing is correct
            line_score[sys][line_i % len(line_src)][metric] = float(score)

    # putting it all together
    data = []
    line_id_true = 0
    for line_i, (line_src, line_ref, line_doc) in enumerate(zip(line_src, line_ref, line_doc)):
        # filter None on the whole row
        if any([line_score[sys][line_i]["score"] in {"None", "0"} for sys in systems]):
        # if any([line_score[sys][line_i]["score"] in {"None"} for sys in systems]):
            continue

        line_domain, line_doc = line_doc.strip().split("\t")

        data.append({
            "i": line_id_true,
            "src": line_src.strip(),
            "ref": line_ref.strip(),
            "tgt": {sys: line_sys[sys][line_i].strip() for sys in systems},
            "domain": line_domain,
            "score": {sys: float(line_score[sys][line_i].pop("score")) for sys in systems},
            "metrics": {sys: line_score[sys][line_i] for sys in systems},
        })
        line_id_true += 1

    if normalize:
        import collections
        data_flat = collections.defaultdict(list)
        for line in data:
            for met_all in line["metrics"].values():
                for met_k, met_v in met_all.items():
                    data_flat[met_k].append(met_v)

            data_flat["score"] += list(line["score"].values())

        # normalize
        data_flat = {
            k: (min(v), max(v))
            for k,v in data_flat.items()
        }

        for line in data:
            for sys, met_all in line["metrics"].items():
                for met_k, met_v in met_all.items():
                    # (x-min)/(max-min) normalize
                    line["metrics"][sys][met_k] = (met_v-data_flat[met_k][0])/(data_flat[met_k][1]-data_flat[met_k][0])

            for sys, sys_v in line["score"].items():
                line["score"][sys]= (sys_v-data_flat["score"][0])/(data_flat["score"][1]-data_flat["score"][0])


    print("Loaded", len(data), "lines of", len(systems), "systems")
    return data

def get_sys_absolute(data_new, metric="score") -> Dict[str, float]:
    import collections
    import numpy as np

    scores_new = collections.defaultdict(list)

    systems = list(data_new[0]["score"].keys())
    if metric == "score":
        for line in data_new:
            for sys, sys_v in line["score"].items():
                scores_new[sys].append(sys_v)
    else:
        for line in data_new:
            for sys in systems:
                scores_new[sys].append(line["metrics"][sys][metric])

    scores_new = {
        sys: np.average(scores_new[sys])
        for sys in systems
    }

    return scores_new

def get_sys_absolute_but_rank(data_new, metric="score"):
    raise Exception("Deprecated")
    import collections
    import numpy as np

    scores_new = collections.defaultdict(list)

    systems = list(data_new[0]["score"].keys())
    if metric == "score":
        for line in data_new:
            scores = sorted(list(line["score"].items()), key=lambda x: x[1])
            for sys_rank, (sys, sys_v) in enumerate(scores):
                scores_new[sys].append(sys_rank)
    else:
        raise NotImplementedError()

    scores_new = {
        sys: np.average(scores_new[sys])
        for sys in systems
    }

    return scores_new

def get_sys_ordering(data_new: list, metric="score"):
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


def eval_data_pairs(data_new: list, data_old: list):
    import itertools
    import numpy as np
    
    systems = list(data_old[0]["score"].keys())

    scores_old = get_sys_absolute(data_old)
    scores_new = get_sys_absolute(data_new)

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

def get_nice_subset(data_old, target_size=100, step_size=10, metric="score"):
    while len(data_old) > target_size:
        order_full = get_sys_ordering(data_old, metric=metric)
        data_old.sort(key=lambda line: get_ord_accuracy(order_full, get_sys_ordering([line], metric=metric)))
        data_old = data_old[step_size:]

    return data_old