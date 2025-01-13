import subset2evaluate.utils as utils
import utils_fig
import numpy as np
import tqdm
import random
from sklearn.preprocessing import StandardScaler


def featurize(line):
    # human oracle
    # scores = np.array(list(line["scores"]["human"].values()))
    # val_median = np.median(scores)
    # return np.abs(scores-val_median)
    return np.array(
        [
            np.average([sys_v["MetricX-23-c"] for sys_v in line["scores"].values()]),
            np.average([sys_v["COMET"] for sys_v in line["scores"].values()]),
            len(line["src"]),
            # len(line["ref"]),
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

    points_y_local = []
    # repeat each sampling 10 times to smooth it out
    for _ in range(2):
        data_prototypes = random.sample(data_old, k=int(len(data_old) * prop))

        data_new = []
        for line in data_old:
            # find nearest prototypical example
            data_new.append(
                min(
                    data_prototypes, key=lambda x: l2_dist(line["feat"], x["feat"])
                )
            )
            # TODO: this is not fair to evaluate with cluster count when the lines are duplicated which adds unfair statistical advantage
            # TODO: should be deduplicated

        # repeat each sampling 10 times to smooth it out
        points_y_local.append(utils.eval_system_clusters(data_new))

    points_y.append(np.average(points_y_local))

print(f"Average  {np.average(points_y):.2f}")

utils_fig.plot_subset_selection([(points_x, points_y, f"k-means {np.average(points_y):.2f}")], "09a-kmeans")
