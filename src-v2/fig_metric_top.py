import utils
import figutils
import numpy as np
import tqdm
import scipy.stats
import itertools

data_old = utils.load_data()

points_x = []
points_y_lo = []
points_y_hi = []

# mre-score-labse-regular', 'MetricX-23', 'chrF', 'COMET', 'f200spBLEU', 'tokengram_F', 'YiSi-1', 'embed_llama', 'XCOMET-XXL', 'BLEU', 'prismRef', 'eBLEU', 'XCOMET-XL', 'MetricX-23-c', 'XCOMET-Ensemble', 'BERTscore', 'XLsim', 'BLEURT-20', 'MetricX-23-b'
def heuristic_abs(line):
    # np.max is also good
    return np.average(
        [sys_v["MetricX-23-c"] for sys_v in line["metrics"].values()]
    )

def heuristic_std(line):
    return np.std([sys_v["MetricX-23"] for sys_v in line["metrics"].values()])

def heuristic_max_diff(line):
    return max([sys["COMET"] for sys in line["metrics"].values()])-min([sys["COMET"] for sys in line["metrics"].values()])

    
def heuristic_moment(line, pow=3):
    # pow = 3: works bets
    # pow = 2: variance
    # pow = 1: abs diff
    return  np.average([
        abs(sys_a_v["MetricX-23"]-sys_b_v["MetricX-23"])**pow
        for sys_a_v, sys_b_v in itertools.combinations(line["metrics"].values(), 2)
    ])

def heuristic_corr(line):
    return scipy.stats.spearmanr(
        [sys_v["COMET"] for sys_v in line["metrics"].values()],
        [sys_v["MetricX-23-c"] for sys_v in line["metrics"].values()]
    )[0]
    

def heuristic_score_abs(line):
    return np.average(list(line["score"].values()))

def heuristic_score_std(line):
    return np.std(list(line["score"].values()))

data_old.sort(key=heuristic_abs)

for prop in tqdm.tqdm(utils.PROPS):
    points_x.append(prop)

    # taking lines with the lowest metric score
    points_y_lo.append(
        utils.eval_data_pairs(data_old[: int(len(data_old) * prop)], data_old)
    )
    points_y_hi.append(
        utils.eval_data_pairs(data_old[-int(len(data_old) * prop) :], data_old)
    )

print(f"Average from lowest  {np.average(points_y_lo):.2%}")
print(f"Average from highest {np.average(points_y_hi):.2%}")


figutils.plot_subsetacc(
    [
        (points_x, points_y_lo, f"From lowest {np.average(points_y_lo):.2%}"),
        (points_x, points_y_hi, f"From highest {np.average(points_y_hi):.2%}"),
    ],
    "metric_top",
)
