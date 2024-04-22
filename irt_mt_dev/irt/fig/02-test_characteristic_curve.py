import random
import irt_mt_dev.utils as utils
import irt_mt_dev.utils.fig
import matplotlib.pyplot as plt
import numpy as np
import json
from irt_mt_dev.irt.basic import IRTModel
import torch

data_wmt = utils.load_data()
# NOTE: no guarantee that this is the same dataset
data_irt = json.load(open("computed/itr_metric.json", "r"))
systems = list(data_irt["systems"].keys())
model = IRTModel.load_from_checkpoint(
    "lightning_logs/version_17/checkpoints/epoch=999-step=150000.ckpt",
    len_items=len(data_wmt),
    systems=systems,
)
irt_mt_dev.utils.fig.matplotlib_default()
plt.figure(figsize=(3, 2))


theta_min = min(data_irt["systems"].values())
theta_max = max(data_irt["systems"].values())
points_x = np.linspace(theta_min-1, theta_max+1, 100)


def predict_item(item, theta):
    return item["c"] + (1 - item["c"]) / (
        1 + np.exp(-item["a"] * (theta - item["b"]))
    )


# plot empirical
plt.scatter(
    x=list(data_irt["systems"].values()),
    y=[np.average([
        x["metrics"][sys]["MetricX-23-c"] > threshold
        for x in data_wmt[1:2]
        # we have 0.1, 0.2, ... 1.0
        for threshold_i, threshold in enumerate(np.linspace(0, 1, 11)[1:])
    ])
        for sys in systems
    ]
)

plt.plot(
    points_x,
    [
        np.average([predict_item(item, theta) for item in data_irt["items"]])
        for theta in points_x
    ],
    color="black"
)

plt.xticks(
    list([x for x in data_irt["systems"].values()]),
    [""]*len(systems),
)

irt_mt_dev.utils.fig.turn_off_spines()
# plt.tight_layout(pad=0.1)
plt.show()
