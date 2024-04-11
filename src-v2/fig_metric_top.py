import utils
import figutils
import numpy as np
import tqdm
import scipy.stats

data_old = utils.load_data()

points_x = []
points_y_lo = []
points_y_hi = []

# mre-score-labse-regular', 'MetricX-23', 'chrF', 'COMET', 'f200spBLEU', 'tokengram_F', 'YiSi-1', 'embed_llama', 'XCOMET-XXL', 'BLEU', 'prismRef', 'eBLEU', 'XCOMET-XL', 'MetricX-23-c', 'XCOMET-Ensemble', 'BERTscore', 'XLsim', 'BLEURT-20', 'MetricX-23-b'

# np.max is also good
data_old.sort(
    key=lambda line: np.average(
        [sys_v["MetricX-23-c"] for sys_v in line["metrics"].values()]
    )
)
# data_old.sort(key=lambda line: np.std([sys_v["MetricX-23"] for sys_v in line["metrics"].values()]))
# data_old.sort(key=lambda line: np.average(list(line["score"].values())))
# data_old.sort(key=lambda line: np.std(list(line["score"].values())))
# data_old.sort(key=lambda line: max([sys["COMET"] for sys in line["metrics"].values()])-min([sys["COMET"] for sys in line["metrics"].values()]))
# data_old.sort(key=lambda line: max([x for x in line["score"].values() if x!=0])-min([x for x in line["score"].values() if x!=0]))
# data_old.sort(key=lambda line: scipy.stats.spearmanr(
#     [sys_v["COMET"] for sys_v in line["metrics"].values()],
#     [sys_v["MetricX-23-c"] for sys_v in line["metrics"].values()]
#     )[0]
# )

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
