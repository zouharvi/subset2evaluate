import utils
import utilsfig
import json
import numpy as np

data_itr = json.load(open("computed/itr_metricx.json", "r"))
data_old = utils.load_data()

systems = list(data_old[0]["score"].keys())
ord_gold = utils.get_sys_ordering(data_old)

# EXPERIMENT: compare ITR MT performance with AVG MT one and with gold one
ord_metricx = utils.get_sys_ordering(data_old, metric="MetricX-23-c")
ord_itr_mt = list(zip(systems, data_itr["mts"]))
ord_itr_mt.sort(key=lambda x: x[1], reverse=True)
ord_itr_mt = {sys: sys_i for sys_i, (sys, sys_v) in enumerate(ord_itr_mt)}

print(f"Acc ITR: {utils.get_ord_accuracy(ord_gold, ord_itr_mt):.2%}")
print(f"Acc MetricX: {utils.get_ord_accuracy(ord_gold, ord_metricx):.2%}")

# EXPERIMENT: take the most difficult/discriminative examples?
points_x = []
points_y_lo = []
points_y_hi = []
for prop in utils.PROPS:
    points_x.append(prop)

    data_old.sort(key=lambda x: data_itr["items"][x["i"]][1])
    
    points_y_lo.append(utils.eval_data_pairs(data_old[:int(len(data_old)*prop)], data_old))
    points_y_hi.append(utils.eval_data_pairs(data_old[-int(len(data_old)*prop):], data_old))
    
print(f"Average from lowest  {np.average(points_y_lo):.2%}")
print(f"Average from highest {np.average(points_y_hi):.2%}")

utilsfig.plot_subsetacc(
    [
        (points_x, points_y_lo, f"From lowest {np.average(points_y_lo):.2%}"),
        (points_x, points_y_hi, f"From highest {np.average(points_y_hi):.2%}"),
    ],
    "irt",
)
