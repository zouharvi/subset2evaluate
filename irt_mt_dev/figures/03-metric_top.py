import irt_mt_dev.utils as utils
import irt_mt_dev.utils.fig
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

def heuristic_translation_dist_chrf(line):
    import sacrebleu
    metric = sacrebleu.metrics.chrf.CHRF()
    out = []
    for text_a, text_b in itertools.product(line["tgt"].values(), repeat=2):
        out.append(metric.sentence_score(hypothesis=text_a, references=[text_b]).score)
    return np.average(out)

def heuristic_translation_dist_unigram(line):
    import collections
    out = []
    for text_a, text_b in itertools.product(line["tgt"].values(), repeat=2):
        text_a = collections.Counter(text_a.split())
        text_b = collections.Counter(text_b.split())
        out.append(2*(text_a & text_b).total()/(text_a.total()+text_b.total()))
    return np.average(out)

# sort by the heuristic
data_old = [(line, heuristic_abs(line)) for line in tqdm.tqdm(data_old)]
data_old.sort(key=lambda x: x[1])
data_old = [x[0] for x in data_old]

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


irt_mt_dev.utils.fig.plot_subsetacc(
    [
        (points_x, points_y_lo, f"From lowest {np.average(points_y_lo):.2%}"),
        (points_x, points_y_hi, f"From highest {np.average(points_y_hi):.2%}"),
    ],
    "metric_top",
)
