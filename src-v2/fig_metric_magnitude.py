import utils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import tqdm

data_old = utils.load_data(langs="cs-uk")

utils.matplotlib_default()
plt.figure(figsize=(3, 2))
points_x = []
points_y = []

# np.max is ok?
data_old.sort(key=lambda line: np.average([sys_v["COMET"] for sys_v in line["metrics"].values()]))
# data_old.sort(key=lambda line: np.corrcoef(
#     [sys_v["COMET"] for sys_v in line["metrics"].values()],
#     [sys_v["MetricX-23"] for sys_v in line["metrics"].values()]
#     )[0,1]
# )

for prop in tqdm.tqdm(np.linspace(0.01, 1, 50)):
    points_x.append(prop)

    # taking lines with the lowest metric score
    points_y.append(utils.eval_data_pairs(data_old[:int(len(data_old)*prop)], data_old))

plt.scatter(
    points_x, points_y,
    marker="o", s=10, label="From lowest"
)
print(f"Average from lowest  {np.average(points_y):.0%}")

points_x = []
points_y = []
for prop in tqdm.tqdm(np.linspace(0.01, 1, 100)):
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
print(f"Average from highest {np.average(points_y):.0%}")


plt.ylabel("Sys. rank accuracy" + " "*5, labelpad=-5)
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