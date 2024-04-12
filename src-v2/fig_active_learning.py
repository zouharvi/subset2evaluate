import copy
import utils
import utilsfig
import random
import numpy as np
import tqdm

data_old = utils.load_data()
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
points_y = []

for prop in tqdm.tqdm(utils.PROPS):
    points_x.append(prop)

    points_y_local = []
    # repeat each sampling 10 times to smooth it out
    for _ in range(10):
        data_old_local = copy.deepcopy(data_old)
        data_new = random.sample(data_old_local, k=10)
        data_new_set_i = {x["i"] for x in data_new}
        data_old_local = [x for x in data_old_local if x["i"] not in data_new_set_i]

        while len(data_new) < int(len(data_old)*prop):
            # this is for purely active learning
            # cur_ord = utils.get_sys_ordering(data_new, metric="score")
            # this is true apriori subset selection
            cur_ord = utils.get_sys_ordering(data_new, metric="MetricX-23-c")

            # TODO: min doesn't make sense here, right? but it works better than max!
            line_lowest_conf = min(data_old_local, key=lambda x: ord_distance(cur_ord, x["ord"]))

            # filter that one from the pool
            # TODO: what if we skip it and keep it there, if it's still "predicted" incorrectly?
            data_new_set_i.add(line_lowest_conf["i"])
            data_old_local = [x for x in data_old_local if x["i"] not in data_new_set_i]

            data_new.append(line_lowest_conf)


        points_y_local.append(utils.eval_data_pairs(data_new, data_old))

    points_y.append(np.average(points_y_local))

print(f"Average  {np.average(points_y):.2%}")
utilsfig.plot_subsetacc([(points_x, points_y, f"{np.average(points_y):.2%}")], "active_learning")