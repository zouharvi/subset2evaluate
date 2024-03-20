import utils
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import tqdm

data_old = utils.load_data()

plt.figure(figsize=(3, 2))
points_x = []
points_y = []

# higher density at the beginning because there are higher changes in y
PROPS = np.concatenate([np.linspace(0, 0.04, 35), np.linspace(0.04, 1, 65)])
for prop in tqdm.tqdm(PROPS):
    points_x.append(prop)

    if prop == 0.0:
        points_y.append(0.5)
    else:
        points_y_local = []
        # repeat each sampling 10 times to smooth it out
        for _ in range(20):
            data_new = random.sample(data_old, k=int(len(data_old)*prop))
            points_y_local.append(utils.eval_data_pairs(data_new, data_old))

        points_y.append(np.average(points_y_local))

plt.scatter(
    points_x, points_y,
    marker="o", s=10, color="black",
)
plt.ylabel("Sys. rank accuracy" + " "*5, labelpad=-5)
plt.xlabel("Proportion of original data", labelpad=-2)

ax = plt.gca()
ax.spines[['top', 'right']].set_visible(False)
ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda y, _: '{:.0%}'.format(y))) 
ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda y, _: '{:.0%}'.format(y))) 

plt.tight_layout(pad=0.1)
plt.savefig("figures/random_subset.png", dpi=200)
plt.savefig("figures/random_subset.pdf")
plt.show()