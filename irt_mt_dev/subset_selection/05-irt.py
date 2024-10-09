import irt_mt_dev.utils as utils
import irt_mt_dev.utils.fig
import json
import numpy as np

data_irt = json.load(open("computed/irt_wmt_4pl_s0_pyirt.json", "r"))[-1]
data_old = utils.load_data()

systems = list(data_old[0]["scores"].keys())
ord_gold = utils.get_sys_ordering(data_old)

# EXPERIMENT: compare ITR MT performance with AVG MT one and with gold one
ord_metricx = utils.get_sys_ordering(data_old, metric="MetricX-23-c")
ord_irt_mt = list(data_irt["systems"].items())
ord_irt_mt.sort(key=lambda x: x[1], reverse=True)
ord_irt_mt = {sys: sys_i for sys_i, (sys, sys_v) in enumerate(ord_irt_mt)}

print(f"Acc ITR: {utils.get_ord_accuracy(ord_gold, ord_irt_mt):.2%}")
print(f"Acc MetricX: {utils.get_ord_accuracy(ord_gold, ord_metricx):.2%}")

# EXPERIMENT: take the most difficult/discriminative examples?
def metric(item_i):
    # aggregared fisher information content
    item = data_irt["items"][item_i["i"]]
    information = 0
    for theta in data_irt["systems"].values():
        prob = utils.pred_irt(
            theta,
            item
        )
        information += prob*(1-prob)*(item["disc"]**2)
    return information

    # return data_irt["items"][item["i"]]["disc"]
    # return data_irt["items"][item["i"]]["feas"]
    # return data_irt["items"][item["i"]]["diff"]

points_x = []
points_y_lo = []
points_y_hi = []
for prop in utils.PROPS:
    points_x.append(prop)

    data_old.sort(key=metric)
    
    points_y_lo.append(utils.eval_data_pairs(data_old[:int(len(data_old)*prop)], data_old))
    points_y_hi.append(utils.eval_data_pairs(data_old[-int(len(data_old)*prop):], data_old))
    
print(f"Average from lowest  {np.average(points_y_lo):.2%}")
print(f"Average from highest {np.average(points_y_hi):.2%}")

irt_mt_dev.utils.fig.plot_subsetacc(
    [
        (points_x, points_y_lo, f"From lowest {np.average(points_y_lo):.2%}"),
        (points_x, points_y_hi, f"From highest {np.average(points_y_hi):.2%}"),
    ],
    "irt",
)
