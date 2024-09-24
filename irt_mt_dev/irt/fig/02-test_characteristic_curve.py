import irt_mt_dev.utils as utils
import irt_mt_dev.utils.fig
import matplotlib.pyplot as plt
import numpy as np
import json

data_wmt = utils.load_data(normalize=True, binarize=False)

# NOTE: no guarantee that this is the same dataset
data_irt = json.load(open("computed/irt_MetricX-23-c_noconst_3pl.json", "r"))
systems = list(data_irt["systems"].keys())
irt_mt_dev.utils.fig.matplotlib_default()
plt.figure(figsize=(3, 2))

theta_min = min(data_irt["systems"].values())
theta_max = max(data_irt["systems"].values())
points_x = np.linspace(theta_min-0.05, theta_max+0.05, 100)

points_y_true = [
    np.average([
        x["scores"][sys]["MetricX-23-c"]
        for x in data_wmt
    ])
    for sys in systems
]
points_y_pred = [
    np.average([
        utils.pred_irt(data_irt["systems"][sys], item)
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
            utils.pred_irt(theta, item)
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
