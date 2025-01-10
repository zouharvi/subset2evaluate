# used for fellowhsip application

import subset2evaluate.utils as utils
import utils_fig
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mtick
import json

data_irt_d0 = json.load(open(f"computed/irt_score_d0.json", "r"))
data_irt_d1 = json.load(open(f"computed/irt_score_d1.json", "r"))
systems = list(data_irt_d0["systems"].keys())
utils_fig.matplotlib_default()
plt.figure(figsize=(2.0, 1.7))


theta_min = min(min(data_irt_d0["systems"].values()), min(data_irt_d1["systems"].values()))
theta_max = max(max(data_irt_d0["systems"].values()), max(data_irt_d1["systems"].values()))
points_x = np.linspace(theta_min-0.4, theta_max+0.4, 100)


points_y_pred_d0 = [
    np.average([
        utils.pred_irt(data_irt_d0["systems"][sys], item)
        for item in data_irt_d0["items"]
    ])
    for sys in systems
]
points_y_pred_d1 = [
    np.average([
        utils.pred_irt(data_irt_d1["systems"][sys], item)
        for item in data_irt_d1["items"]
    ])
    for sys in systems
]


plt.plot(
    [
        np.average([
            utils.pred_irt(theta, item)
            for item in data_irt_d0["items"]
        ])
        for theta in points_x
    ],
    [
        np.average([
            utils.pred_irt(theta, item)
            for item in data_irt_d1["items"]
        ])
        for theta in points_x
    ],
    color="black",
    linewidth=3
)

plt.ylim(0.65, 1.0)
plt.xlim(0.65, 1.0)
plt.ylabel("Metric score\n(easy set)", labelpad=-2)
plt.xlabel("Metric score (hard set)" + " " * 5)
plt.xticks([0.7, 0.8, 0.9, 1.0])

plt.hlines(y=0.89, xmin=0.6, xmax=0.7, zorder=-1)
plt.vlines(x=0.7, ymin=0.6, ymax=0.89, zorder=-1)
plt.text(
    x=0.87, y=0.7,
    s="70 on hard set\n=\n89 on easy set",
    ha="center",
    fontsize=9,
    color="#582f7a"
)


plt.gca().xaxis.set_major_formatter(mtick.FuncFormatter(lambda y, _: f"{y*100:.0f}")) 
plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(lambda y, _: f"{y*100:.0f}")) 

utils_fig.turn_off_spines()
plt.tight_layout(pad=0)
plt.savefig(f"figures/comparable_theta_joint.pdf")
plt.show()
