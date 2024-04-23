import irt_mt_dev.utils as utils
import irt_mt_dev.utils.fig
import matplotlib.pyplot as plt
import numpy as np
import json

data_wmt = utils.load_data(normalize=True)
# data_wmt = utils.get_nice_subset(data_wmt, target_size=100, step_size=50, metric="score")


# NOTE: no guarantee that this is the same dataset
data_irt = json.load(open("computed/itr_human.json", "r"))
systems = list(data_irt["systems"].keys())
irt_mt_dev.utils.fig.matplotlib_default()
plt.figure(figsize=(3, 2))


theta_min = min(data_irt["systems"].values())
theta_max = max(data_irt["systems"].values())
points_x = np.linspace(theta_min, theta_max, 100)


def predict_item(item, theta):
    return item["a"] * theta
    return item["c"] + (1 - item["c"]) / (
        1 + np.exp(-item["a"] * (theta - item["b"]))
    )


# plot empirical
plt.scatter(
    x=list(data_irt["systems"].values()),
    y=[np.average([
        x["score"][sys]
        for x_i, x in enumerate(data_wmt)
        # if data_irt["items"][x_i]["a"] >0
    ])
        for sys in systems
    ]
)

system_scores = [
    (
        sys,
        np.average([
            line["score"][sys]
            for line in data_wmt
        ]))
    for sys in systems
]
system_scores.sort(key=lambda x: x[1])
print("system average", system_scores)

plt.plot(
    points_x,
    [
        np.average([
            predict_item(item, theta)
            for item in data_irt["items"]
            # if item["a"] >0
        ])
        for theta in points_x
    ],
    color="black"
)

plt.xticks(
    list([x for x in data_irt["systems"].values()]),
    [""]*len(systems),
)
plt.xlabel(r"$\theta$ (systems)")
plt.ylabel("Expected performance")

irt_mt_dev.utils.fig.turn_off_spines()
plt.tight_layout(pad=0.1)
plt.show()
