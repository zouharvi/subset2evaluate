import subset2evaluate.utils as utils
import utils_fig
import matplotlib.pyplot as plt
import random
import numpy as np
import argparse

args = argparse.ArgumentParser()
args.add_argument("metric", default="MetricX-23")
args = args.parse_args()

data_old = utils.load_data_wmt(normalize=True)
utils_fig.matplotlib_default()
plt.figure(figsize=(3, 2))

def renormalize(data_y):
    data_y = np.array(data_y)
    min_y, max_y = np.quantile(data_y, [0.01, 0.99])
    data_y = np.clip(data_y, min_y, max_y)

    data_y = (data_y - min_y) / (max_y - min_y)
    mean = np.mean(data_y)
    
    return data_y
    # offset = 0.5 - mean
    # return [y+offset for y in data_y]


systems = list(data_old[0]["scores"].keys())
points_metric = [
    sent["scores"][sys][args.metric]
    for sent in data_old
    for sys in systems
]
# plt.hist(
#     points_score,
#     bins=np.linspace(0, 1, 30),
#     label="Human", alpha=0.5
# )

plt.hist(
    points_metric,
    bins=np.linspace(0, 1, 30),
    label="Metric", alpha=0.5
)
plt.hist(
    renormalize(points_metric),
    bins=np.linspace(0, 1, 30),
    label="Metric (renorm.)",
    alpha=0.5,
)

plt.xlabel("Score", labelpad=-10)
plt.xticks([0, 0.25, 0.75, 1.0])
plt.ylabel("Frequency")
plt.yticks([])

utils_fig.turn_off_spines()
plt.tight_layout(pad=0.1)
plt.legend()
plt.show()
