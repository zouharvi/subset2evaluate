PROPS = [x/100 for x in range(10, 90+1, 10)]

def load_data(year="wmt23", langs="en-cs"):
    import glob
    import collections
    line_src = open(f"data/mt-metrics-eval-v2/{year}/sources/{langs}.txt", "r").readlines()
    line_ref = open(f"data/mt-metrics-eval-v2/{year}/references/{langs}.refA.txt", "r").readlines()
    line_sys = {}
    for f in glob.glob(f"data/mt-metrics-eval-v2/{year}/system-outputs/{langs}/*.txt"):
        sys = f.split("/")[-1].removesuffix(".txt")
        if sys in {"synthetic_ref", "refA"}:
                continue

        line_sys[sys] = open(f, "r").readlines()
    systems = list(line_sys.keys())

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
    for line_i, (line_src, line_ref) in enumerate(zip(line_src, line_ref)):
        # filter None on the whole row
        # if any([line_score[sys][line_i]["score"] in {"None", "0"} for sys in systems]):
        if any([line_score[sys][line_i]["score"] in {"None"} for sys in systems]):
            continue

        data.append({
            "i": line_id_true,
            "src": line_src.strip(),
            "ref": line_ref.strip(),
            "tgt": {sys: line_sys[sys][line_i].strip() for sys in systems},
            "score": {sys: float(line_score[sys][line_i].pop("score")) for sys in systems},
            "metrics": {sys: line_score[sys][line_i] for sys in systems},
        })
        line_id_true += 1

    print("Loaded", len(data), "lines of", len(systems), "systems")
    return data

COLORS = [
    "#bc272d", # red
    "#50ad9f", # green
    "#0000a2", # blue
    "#e9c716", # yellow
]

def matplotlib_default():
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.rcParams["axes.prop_cycle"] = plt.cycler(color=COLORS)

def get_sys_absolute(data_new, metric="score"):
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