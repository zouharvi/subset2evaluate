import utils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import tqdm
import random
from sklearn.preprocessing import StandardScaler


def featurize(line):
    # human oracle
    # scores = np.array(list(line["score"].values()))
    # val_median = np.median(scores)
    # return np.abs(scores-val_median)
    
    return np.array(
        [
            np.max([sys_v["COMET"] for sys_v in line["metrics"].values()]),
            len(line["src"]),
            len(line["ref"]),
        ]
    )

def l2_dist(a, b):
    return np.linalg.norm(a - b)


data_old = utils.load_data()
data_old_vec = StandardScaler().fit_transform([featurize(line) for line in data_old])
for line, line_feat in zip(data_old, data_old_vec):
    line["feat"] = line_feat
del data_old_vec


utils.matplotlib_default()
plt.figure(figsize=(3, 2))
points_x = []
points_y = []

for prop in tqdm.tqdm(utils.PROPS):
    points_x.append(prop)


    points_y_local = []
    # repeat each sampling 10 times to smooth it out
    for _ in range(10):
        data_prototypes = random.sample(data_old, k=int(len(data_old) * prop))

        data_new = []
        for line in data_old:
            # find nearest prototypical example
            data_new.append(
                min(
                    data_prototypes, key=lambda x: l2_dist(line["feat"], x["feat"])
                )
            )

        # repeat each sampling 10 times to smooth it out
        points_y_local.append(utils.eval_data_pairs(data_new, data_old))

    points_y.append(np.average(points_y_local))

print(f"Average  {np.average(points_y):.2%}")


plt.scatter(
    points_x,
    points_y,
    marker="o",
    s=10,
    color="black",
)
plt.ylabel("Sys. rank accuracy" + " " * 5, labelpad=-5)
plt.xlabel("Proportion of original data", labelpad=-2)

ax = plt.gca()
ax.spines[['top', 'right']].set_visible(False)
ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda y, _: f'{y:.0%}'))
ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda y, _: f'{y:.0%}'))

plt.ylim(0.7, 1)
plt.tight_layout(pad=0.1)
plt.savefig("figures-v2/kmeans_fake.png", dpi=200)
plt.savefig("figures-v2/kmeans_fake.pdf")
plt.show()
