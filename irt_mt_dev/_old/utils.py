def load_data():
    import pickle
    return pickle.load(open("data/toship21.simple.pkl", "rb"))

def eval_data_pairs(data_new, data_old):
    import itertools
    import collections
    import numpy as np

    scores_old = collections.defaultdict(list)
    scores_new = collections.defaultdict(list)
    for line in data_old:
        for sys, sys_v in line.items():
            scores_old[sys].append(sys_v["scores"]["human"])
    for line in data_new:
        for sys, sys_v in line.items():
            scores_new[sys].append(sys_v["scores"]["human"])

    systems = {
        sys
        for line in data_old
        for sys in line.keys()
    }

    if len(scores_new) < len(scores_old):
        print("WARNING: Some systems don't have a single line.")

    scores_old = {
        sys: np.average(scores_old[sys])
        for sys in systems
    }
    scores_new = {
        # default to 0 if we dropped all lines for a system
        sys: np.average(scores_new[sys]) if sys in scores_new else 0
        for sys in systems
    }

    result = []
    for sys1, sys2 in itertools.combinations(systems, 2):
        result.append((scores_old[sys1]<scores_old[sys2])==(scores_new[sys1]<scores_new[sys2]))

    return np.average(result)
    
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