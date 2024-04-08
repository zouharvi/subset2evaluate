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

# higher density at the beginning because there are higher changes in y
PROPS = np.concatenate([np.linspace(0, 0.04, 35), np.linspace(0.04, 1, 65)])

data_old.sort(key=lambda line: np.corrcoef([sys_v["metrics"]["metric_COMET"] for sys_v in line.values()], [sys_v["metrics"]["metric_SacreBLEU_bleu"] for sys_v in line.values()])[0,1])
for prop in tqdm.tqdm(PROPS):
    points_x.append(prop)

    if prop == 0.0:
        points_y.append(0.5)
    else:
        # taking lines with the lowest metric score
        points_y.append(utils.eval_data_pairs(data_old[:int(len(data_old)*prop)], data_old))

plt.scatter(
    points_x, points_y,
    marker="o", s=10, label="From lowest"
)

points_x = []
points_y = []
for prop in tqdm.tqdm(PROPS):
    points_x.append(prop)

    if prop == 0.0:
        points_y.append(0.5)
    else:
        # taking lines with the highest metric score
        points_y.append(utils.eval_data_pairs(data_old[-int(len(data_old)*prop):], data_old))

plt.scatter(
    points_x, points_y,
    marker="o", s=10, label="From highest"
)


plt.ylabel("Sys. rank accuracy" + " "*5, labelpad=-5)
plt.xlabel("Proportion of original data", labelpad=-2)

plt.legend(frameon=False)

ax = plt.gca()
ax.spines[['top', 'right']].set_visible(False)
ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda y, _: '{:.0%}'.format(y))) 
ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda y, _: '{:.0%}'.format(y))) 

plt.tight_layout(pad=0.1)
plt.savefig("figures-v1metric_disagree.png", dpi=200)
plt.savefig("figures-v1metric_disagree.pdf")
plt.show()