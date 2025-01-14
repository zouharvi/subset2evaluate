import subset2evaluate.utils as utils
import utils_fig
import numpy as np
import tqdm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import subset2evaluate.evaluate


def featurize(line):
    return np.array(
        [
            np.average([sys_v["COMET"] for sys_v in line["scores"].values()]),
            # np.average([sys_v["MetricX-23"] for sys_v in line["scores"].values()]),
            np.var([sys_v["COMET"] for sys_v in line["scores"].values()]),
            # np.var([sys_v["MetricX-23"] for sys_v in line["scores"].values()]),
            len(line["src"]),
            len(line["ref"]),
        ]
    )


def l2_dist(a, b):
    return np.linalg.norm(a - b)


data_old = utils.load_data_wmt()
data_old_vec = StandardScaler().fit_transform([featurize(line) for line in data_old])
for line, line_feat in zip(data_old, data_old_vec):
    line["feat"] = line_feat
del data_old_vec

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
        # TODO: this is not fair to evaluate with cluster count when the lines are duplicated which adds unfair statistical advantage
        # TODO: ;should be deduplicated

    # duplicate each example based on its nearest prototype
    data_new = []
    for cluster_i in kmeans.labels_:
        data_new.append(cluster_prototypes[cluster_i])

    # repeat each sampling 10 times to smooth it out
    points_y.append(subset2evaluate.evaluate.eval_subset_clusters(data_new))

print(f"Average  {np.average(points_y):.2f}")

utils_fig.plot_subset_selection([(points_x, points_y, f"k-means true {np.average(points_y):.2f}")], "09c-kmeans_true")
