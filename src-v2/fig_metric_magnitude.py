import utils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import tqdm

data_old = utils.load_data()

utils.matplotlib_default()
plt.figure(figsize=(3, 2))
points_x = []
points_y = []

# mre-score-labse-regular', 'MetricX-23', 'chrF', 'COMET', 'f200spBLEU', 'tokengram_F', 'YiSi-1', 'embed_llama', 'XCOMET-XXL', 'BLEU', 'prismRef', 'eBLEU', 'XCOMET-XL', 'MetricX-23-c', 'XCOMET-Ensemble', 'BERTscore', 'XLsim', 'BLEURT-20', 'MetricX-23-b'

# np.max is also good
data_old.sort(key=lambda line: np.average([sys_v["MetricX-23-c"] for sys_v in line["metrics"].values()]))
# data_old.sort(key=lambda line: np.std([sys_v["MetricX-23"] for sys_v in line["metrics"].values()]))
# data_old.sort(key=lambda line: np.average(list(line["score"].values())))
# data_old.sort(key=lambda line: np.std(list(line["score"].values())))
# data_old.sort(key=lambda line: max([sys["COMET"] for sys in line["metrics"].values()])-min([sys["COMET"] for sys in line["metrics"].values()]))
# data_old.sort(key=lambda line: max([x for x in line["score"].values() if x!=0])-min([x for x in line["score"].values() if x!=0]))
# data_old.sort(key=lambda line: np.corrcoef(
#     [sys_v["COMET"] for sys_v in line["metrics"].values()],
#     [sys_v["MetricX-23-c"] for sys_v in line["metrics"].values()]
#     )[0,1]
# )

for prop in tqdm.tqdm(utils.PROPS):
    points_x.append(prop)

    # taking lines with the lowest metric score
    points_y.append(
        utils.eval_data_pairs(data_old[: int(len(data_old) * prop)], data_old)
    )

plt.scatter(points_x, points_y, clip_on=False, marker="o", s=10, label="From lowest")
print(f"Average from lowest  {np.average(points_y):.2%}")

points_x = []
points_y = []
for prop in tqdm.tqdm(utils.PROPS):
    points_x.append(prop)

    # taking lines with the highest metric score
    points_y.append(
        utils.eval_data_pairs(data_old[-int(len(data_old) * prop) :], data_old)
    )

plt.scatter(points_x, points_y, clip_on=False, marker="o", s=10, label="From highest")
print(f"Average from highest {np.average(points_y):.2%}")


plt.ylabel("Sys. rank accuracy" + " " * 5, labelpad=-5)
plt.xlabel("Proportion of original data", labelpad=-2)

plt.legend(frameon=False, handletextpad=-0.2)

ax = plt.gca()
ax.spines[['top', 'right']].set_visible(False)
ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

plt.ylim(0.7, 1)
plt.tight_layout(pad=0.1)
plt.savefig("figures-v2/metric_magnitude.png", dpi=200)
plt.savefig("figures-v2/metric_magnitude.pdf")
plt.show()
