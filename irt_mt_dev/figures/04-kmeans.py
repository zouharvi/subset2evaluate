import utils
import utils.fig
import numpy as np
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
            np.max([sys_v["MetricX-23-c"] for sys_v in line["metrics"].values()]),
            np.max([sys_v["COMET"] for sys_v in line["metrics"].values()]),
            len(line["src"]),
            # len(line["ref"]),
        ]
    )

def l2_dist(a, b):
    return np.linalg.norm(a - b)


data_old = utils.load_data()
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

        # repeat each sampling 10 times to smooth it out
        points_y_local.append(utils.eval_data_pairs(data_new, data_old))

    points_y.append(np.average(points_y_local))

print(f"Average  {np.average(points_y):.2%}")

utils.fig.plot_subsetacc([(points_x, points_y, f"{np.average(points_y):.2%}")], "kmeans")
