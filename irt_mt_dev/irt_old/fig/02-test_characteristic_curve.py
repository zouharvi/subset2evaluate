import irt_mt_dev.utils as utils
import irt_mt_dev.utils.fig
import matplotlib.pyplot as plt
import numpy as np
import json

data_wmt = utils.load_data(normalize=True)

# NOTE: no guarantee that this is the same dataset
data_irt = json.load(open("computed/irt_metric.json", "r"))
systems = list(data_irt["systems"].keys())
irt_mt_dev.utils.fig.matplotlib_default()
plt.figure(figsize=(3, 2))


theta_min = min(data_irt["systems"].values())
theta_max = max(data_irt["systems"].values())
points_x = np.linspace(theta_min-0.2, theta_max+0.2, 100)


def predict_item(item, theta):
    return 1 / (1 + np.exp(-item["a"] * (theta - item["b"])))


_median = np.median([
    line["metrics"][sys]["MetricX-23-c"]
    for line in data_wmt
    for sys in systems
])

points_y_true = [
    np.average([
        x["metrics"][sys]["MetricX-23-c"]
        # x["metrics"][sys]["MetricX-23-c"] > _median
        for x in data_wmt
    ])
    for sys in systems
]
points_y_pred = [
    np.average([
        predict_item(item, data_irt["systems"][sys])
        for item in data_irt["items"]
    ])
    for sys in systems
]
print(f"Correlation: {np.corrcoef(points_y_true, points_y_pred)[0,1]:.2%}")

# plot empirical
plt.scatter(
    x=list(data_irt["systems"].values()),
    y=points_y_true
)

plt.plot(
    points_x,
    [
        np.average([
            predict_item(item, theta)
            for item in data_irt["items"]
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
