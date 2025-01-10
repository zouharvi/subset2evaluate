import subset2evaluate.utils as utils
import utils_fig
import matplotlib.pyplot as plt
import numpy as np
import json

data_wmt = utils.load_data_wmt(normalize=True, binarize=False)

data_irt = json.load(
    open("computed/irt_wmt_4pl_s0_eall_metricx.json", "r"))[10]
systems_i = list(range(len(data_irt["theta"])))
utils_fig.matplotlib_default()
plt.figure(figsize=(3, 2))

theta_min = min(data_irt["theta"])
theta_max = max(data_irt["theta"])
points_x = np.linspace(theta_min-0.05, theta_max+0.05, 100)

points_y_true = [
    np.average([
        list(x["scores"].values())[sys_i]["MetricX-23-c"]
        for x in data_wmt
    ])
    for sys_i in systems_i
]
points_y_pred = [
    np.average([
        utils.pred_irt(
            data_irt["theta"][sys_i], {
            "disc": disc,
            "diff": diff,
            "feas": feas,
        })
        for disc, diff, feas in zip(data_irt["disc"], data_irt["diff"], data_irt["feas"])
    ])
    for sys_i in systems_i
]
print(f"Correlation: {np.corrcoef(points_y_true, points_y_pred)[0,1]:.2%}")

# plot empirical
plt.scatter(
    x=data_irt["theta"],
    y=points_y_pred,
    zorder=10,
    label="IRT predicted"
)
plt.scatter(
    x=data_irt["theta"],
    y=points_y_true,
    zorder=10,
    label="True"
)

plt.plot(
    points_x,
    [
        np.average([
            utils.pred_irt(theta, {
                "disc": disc,
                "diff": diff,
                "feas": feas,
            })
            for disc, diff, feas in zip(data_irt["disc"], data_irt["diff"], data_irt["feas"])
        ])
        for theta in points_x
    ],
    color="black"
)

plt.xticks(
    list([x for x in data_irt["theta"]]),
    [""]*len(systems_i),
)
plt.xlabel(r"$\theta$ (systems)")
plt.ylabel("Expected performance")
plt.legend(handletextpad=0.1)

utils_fig.turn_off_spines()
plt.tight_layout(pad=0.1)
plt.show()
