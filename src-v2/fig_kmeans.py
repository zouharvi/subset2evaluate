import utils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import tqdm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def featurize(line):
    return np.array(
        [
            # np.average([sys_v["COMET"] for sys_v in line["metrics"].values()]),
            np.average([sys_v["MetricX-23"] for sys_v in line["metrics"].values()]),
            # np.var([sys_v["COMET"] for sys_v in line["metrics"].values()]),
            # np.var([sys_v["MetricX-23"] for sys_v in line["metrics"].values()]),
            # len(line["src"]),
            # len(line["ref"]),
        ]
    )


def l2_dist(a, b):
    return np.linalg.norm(a - b)


data_old = utils.load_data(langs="de-en")
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

    kmeans = KMeans(
        n_clusters=int(len(data_old) * prop), random_state=0, n_init="auto"
    ).fit([line["feat"] for line in data_old])

    cluster_prototypes = []
    for cluster_center_feat in kmeans.cluster_centers_:
        # find cluster prototypical example
        cluster_prototypes.append(
            min(
                data_old, key=lambda x: l2_dist(cluster_center_feat, x["feat"])
            )
        )

    # duplicate each example based on its nearest prototype
    data_new = []
    for cluster_i in kmeans.labels_:
        data_new.append(cluster_prototypes[cluster_i])

    # repeat each sampling 10 times to smooth it out
    points_y.append(utils.eval_data_pairs(data_new, data_old))

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
plt.savefig("figures-v2/kmeans.png", dpi=200)
plt.savefig("figures-v2/kmeans.pdf")
plt.show()
