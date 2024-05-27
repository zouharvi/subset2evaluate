import irt_mt_dev.utils as utils
import irt_mt_dev.utils.fig
import matplotlib.pyplot as plt
import numpy as np
import json
import argparse

args = argparse.ArgumentParser()
args.add_argument("suffix", default="d0")
args = args.parse_args()

data_irt = json.load(open(f"computed/irt_score_{args.suffix}.json", "r"))
systems = list(data_irt["systems"].keys())
irt_mt_dev.utils.fig.matplotlib_default()
plt.figure(figsize=(1.7, 1.7))


theta_min = 0.8
theta_max = 2.7
points_x = np.linspace(theta_min-0.2, theta_max+0.2, 100)


def predict_item(item, theta):
    return 1 / (1 + np.exp(-item["a"] * (theta - item["b"])))


points_y_pred = [
    np.average([
        predict_item(item, data_irt["systems"][sys])
        for item in data_irt["items"]
    ])
    for sys in systems
]


plt.plot(
    points_x,
    [
        np.average([
            predict_item(item, theta)
            for item in data_irt["items"]
        ])
        for theta in points_x
    ],
    color="black",
    linewidth=2.5,
)

plt.ylim(0.65, 1.0)
plt.xlim(0, 3)
plt.xticks([0, 3], [0, 3])
plt.yticks([0.65, 1.0], ["65", "100"])
plt.xlabel(r"$\theta$ (ability)", labelpad=-10)
plt.ylabel("Metric score\n" + {"d0": "(easy set)", "d1": "(hard set)"}[args.suffix], labelpad=-15)

irt_mt_dev.utils.fig.turn_off_spines()
plt.tight_layout(pad=0.0)
plt.savefig(f"figures/comparable_theta_{args.suffix}.pdf")
plt.show()
