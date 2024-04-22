import irt_mt_dev.utils as utils
import irt_mt_dev.utils.fig
import matplotlib.pyplot as plt
import numpy as np
import json

data_wmt = utils.load_data(normalize=True)
# NOTE: no guarantee that this is the same dataset
data_irt = json.load(open("computed/itr_human.json", "r"))
systems = list(data_irt["systems"].keys())
irt_mt_dev.utils.fig.matplotlib_default()
plt.figure(figsize=(3, 2))


theta_min = min(data_irt["systems"].values())
theta_max = max(data_irt["systems"].values())
points_x = np.linspace(theta_min-5, theta_max+5, 100)


def predict_item(item, theta):
    return item["c"] + (1 - item["c"]) / (
        1 + np.exp(-item["a"] * (theta - item["b"]))
    )


I = 10

# plot empirical
plt.scatter(
    x=list(data_irt["systems"].values()),
    y=[np.average([
        x["score"][sys]
        for x in data_wmt[I:I+1]
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
        np.average([predict_item(item, theta)
                   for item in data_irt["items"][I:I+1]])
        for theta in points_x
    ],
    color="black"
)

plt.xticks(
    list([x for x in data_irt["systems"].values()]),
    [""]*len(systems),
)

irt_mt_dev.utils.fig.turn_off_spines()
plt.tight_layout(pad=0.1)
plt.show()
