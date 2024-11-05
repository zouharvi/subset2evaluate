import irt_mt_dev.utils as utils
import irt_mt_dev.utils.fig
import json
import numpy as np

data_irt = json.load(open("computed/irt_wmt_4pl_s0_pyirt.json", "r"))[-1]
data_old = utils.load_data_wmt()

systems = list(data_old[0]["scores"].keys())
ord_gold = utils.get_sys_ordering(data_old)

# EXPERIMENT: compare ITR MT performance with AVG MT one and with gold one
ord_metricx = utils.get_sys_ordering(data_old, metric="MetricX-23-c")
ord_irt_mt = list(data_irt["systems"].items())
ord_irt_mt.sort(key=lambda x: x[1], reverse=True)
ord_irt_mt = {sys: sys_i for sys_i, (sys, sys_v) in enumerate(ord_irt_mt)}

def utility_information_content(item):
    # aggregared fisher information content
    item = data_irt["items"][item["i"]]

    information = 0
    for theta in data_irt["systems"].values():
        prob = utils.pred_irt(
            theta,
            item
        )
        information += prob*(1-prob)*(item["disc"]**2)
    return information


def utility_diffdisc(item):
    item = data_irt["items"][item["i"]]
    return item["diff"] * item["disc"]

def utility_feas(item):
    item = data_irt["items"][item["i"]]
    return item["feas"]

points_x = []
points_y_diffdisc_acc = []
points_y_diffdisc_clu = []
points_y_infocontent_acc = []
points_y_infocontent_clu = []
points_y_feas_acc = []
points_y_feas_clu = []

data_irt_diffdisc = sorted(data_old, key=lambda x: -utility_diffdisc(x))
data_irt_infocontent = sorted(data_old, key=lambda x: -utility_information_content(x))
data_irt_feas = sorted(data_old, key=lambda x: utility_feas(x))

for prop in utils.PROPS:
    points_x.append(prop)

    points_y_diffdisc_acc.append(utils.eval_order_accuracy(data_irt_diffdisc[:int(len(data_old)*prop)], data_old))
    points_y_diffdisc_clu.append(utils.eval_system_clusters(data_irt_diffdisc[:int(len(data_old)*prop)]))
    points_y_infocontent_acc.append(utils.eval_order_accuracy(data_irt_infocontent[:int(len(data_old)*prop)], data_old))
    points_y_infocontent_clu.append(utils.eval_system_clusters(data_irt_infocontent[:int(len(data_old)*prop)]))
    points_y_feas_acc.append(utils.eval_order_accuracy(data_irt_feas[:int(len(data_old)*prop)], data_old))
    points_y_feas_clu.append(utils.eval_system_clusters(data_irt_feas[:int(len(data_old)*prop)]))

irt_mt_dev.utils.fig.plot_subset_selection(
    [
        (points_x, points_y_feas_acc, f"IRT feasability {np.average(points_y_feas_acc):.2%}"),
        (points_x, points_y_diffdisc_acc, f"IRT difficulty$\\times$discrim. {np.average(points_y_diffdisc_acc):.2%}"),
        (points_x, points_y_infocontent_acc, f"IRT information content {np.average(points_y_infocontent_acc):.2%}"),
    ],
    "12-irt_all",
)
irt_mt_dev.utils.fig.plot_subset_selection(
    [
        (points_x, points_y_feas_clu, f"IRT feasability {np.average(points_y_feas_clu):.2f}"),
        (points_x, points_y_diffdisc_clu, f"IRT difficulty$\\times$discrim. {np.average(points_y_diffdisc_clu):.2f}"),
        (points_x, points_y_infocontent_clu, f"IRT information content {np.average(points_y_infocontent_clu):.2f}"),
    ],
    "12-irt_all",
)
