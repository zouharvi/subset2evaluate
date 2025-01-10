import copy
import subset2evaluate.utils as utils
import utils_fig
import random
import numpy as np
import tqdm

data_old = utils.load_data_wmt()
for line in data_old:
    # TODO: also try with oracle?
    line["ord"] = utils.get_sys_ordering([line], metric="MetricX-23-c")

def ord_distance(ord_a: dict, ord_b: dict):
    return np.average([
        # np.abs(ord_a[sys]-ord_b[sys])
        np.square(ord_a[sys]-ord_b[sys])
        for sys in ord_a.keys()
    ])

points_x = []
points_y_lo = []
points_y_hi = []

for prop in tqdm.tqdm(utils.PROPS):
    points_x.append(prop)

    points_y_lo_local = []
    points_y_hi_local = []
    # repeat each sampling 10 times to smooth it out
    for _ in range(10):
        data_old_lo_local = copy.deepcopy(data_old)
        data_old_hi_local = copy.deepcopy(data_old)

        data_new_lo = random.sample(data_old_lo_local, k=10)
        data_new_hi = random.sample(data_old_hi_local, k=10)
        data_new_lo_set_i = {x["i"] for x in data_new_lo}
        data_new_hi_set_i = {x["i"] for x in data_new_lo}
        data_old_lo_local = [x for x in data_old_lo_local if x["i"] not in data_new_lo_set_i]
        data_old_hi_local = [x for x in data_old_hi_local if x["i"] not in data_new_hi_set_i]

        while len(data_new_lo) < int(len(data_old)*prop):
            # this is for purely active learning
            # cur_ord = utils.get_sys_ordering(data_new, metric="score")
            # this is true apriori subset selection
            cur_ord = utils.get_sys_ordering(data_new_lo, metric="MetricX-23-c")

            # min doesn't make sense here, right? but it works better than max!
            line_lo_conf = min(data_old_lo_local, key=lambda x: ord_distance(cur_ord, x["ord"]))

            # filter that one from the pool
            # TODO: what if we skip it and keep it there, if it's still "predicted" incorrectly?
            data_new_lo_set_i.add(line_lo_conf["i"])
            data_old_lo_local = [x for x in data_old_lo_local if x["i"] not in data_new_lo_set_i]
            data_new_lo.append(line_lo_conf)

        while len(data_new_hi) < int(len(data_old)*prop):
            cur_ord = utils.get_sys_ordering(data_new_hi, metric="MetricX-23-c")
            line_hi_conf = max(data_old_hi_local, key=lambda x: ord_distance(cur_ord, x["ord"]))
            data_new_hi_set_i.add(line_hi_conf["i"])
            data_old_hi_local = [x for x in data_old_hi_local if x["i"] not in data_new_hi_set_i]
            data_new_hi.append(line_hi_conf)

        points_y_lo_local.append(utils.eval_system_clusters(data_new_lo))
        points_y_hi_local.append(utils.eval_system_clusters(data_new_hi))

    points_y_lo.append(np.average(points_y_lo_local))
    points_y_hi.append(np.average(points_y_hi_local))

print(f"Average (lo) {np.average(points_y_lo):.2f}")
print(f"Average (hi) {np.average(points_y_hi):.2f}")

utils_fig.plot_subset_selection(
    [
        (points_x, points_y_lo, f"From closest {np.average(points_y_lo):.2f}"),
        (points_x, points_y_hi, f"From different {np.average(points_y_hi):.2f}"),
    ],
    "10-decision_boundary",
)
