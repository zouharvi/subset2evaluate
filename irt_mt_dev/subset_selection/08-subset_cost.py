import irt_mt_dev.utils as utils
import irt_mt_dev.utils.fig
import json
import numpy as np
import matplotlib.pyplot as plt

irt_mt_dev.utils.fig.matplotlib_default()

data_irt = json.load(open("computed/irt_wmt_4pl_s0_pyirt.json", "r"))[-1]
data_old = utils.load_data()

def metric(item):
    # aggregared fisher information content
    item = data_irt["items"][item["i"]]

    # alternatives
    # return item["disc"]
    # return item["feas"]
    # return item["diff"]
    # return item["diff"]*item["disc"]

    information = 0
    for theta in data_irt["systems"].values():
        prob = utils.pred_irt(
            theta,
            item
        )
        information += prob*(1-prob)*(item["disc"]**2)
    return information


data_old = [
    (
        len(item["src"].split()),
        metric(item)
    )
    for item in data_old
]
# sort by cost
data_old.sort(key=lambda x: x[0], reverse=True)

plt.scatter(
    [x[0] for x in data_old],
    [x[1] for x in data_old],
    alpha=0.2,
    linewidth=0,
)
plt.ylabel("Information content")
plt.xlabel("Sentence length")
corr = np.corrcoef([x[0] for x in data_old], [x[1] for x in data_old])[0, 1]
plt.title(f"IRT model information content vs. sentence length, $\\rho = {corr:.4f}$")
plt.show()