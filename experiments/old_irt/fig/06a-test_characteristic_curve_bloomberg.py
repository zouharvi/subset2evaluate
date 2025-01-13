# used for fellowhsip application

import subset2evaluate.utils as utils
import utils_fig
import matplotlib.pyplot as plt
import numpy as np
import json
import argparse

args = argparse.ArgumentParser()
args.add_argument("suffix", default="d0")
args = args.parse_args()

data_irt = json.load(open(f"computed/irt_score_{args.suffix}.json", "r"))
systems = list(data_irt["systems"].keys())
utils_fig.matplotlib_default()
plt.figure(figsize=(1.7, 1.7))


theta_min = 0.8
theta_max = 2.7
points_x = np.linspace(theta_min - 0.2, theta_max + 0.2, 100)


points_y_pred = [
    np.average([
        utils.pred_irt(data_irt["systems"][sys], item)
        for item in data_irt["items"]
    ])
    for sys in systems
]


plt.plot(
    points_x,
    [
        np.average([
            utils.pred_irt(theta, item)
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
plt.yticks([0.7, 1.0], ["70", "100"])
plt.xlabel(r"$\theta$ (ability)", labelpad=-10)
plt.ylabel("Metric score\n" + {"d0": "(hard set)", "d1": "(easy set)"}[args.suffix], labelpad=-15)

utils_fig.turn_off_spines()
plt.tight_layout(pad=0.0)
plt.savefig(f"figures/comparable_theta_{args.suffix}.pdf")
plt.show()
