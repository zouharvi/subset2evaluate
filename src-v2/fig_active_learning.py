import copy
import utils
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import tqdm

data_old = utils.load_data()
for line in data_old:
    # TODO: also try with oracle?
    line["ord"] = utils.get_sys_ordering([line], metric="MetricX-23-c")

def ord_distance(ord_a: dict, ord_b: dict):
    return np.average([
        np.abs(ord_a[sys]-ord_b[sys])
        # np.square(ord_a[sys]-ord_b[sys])
        for sys in ord_a.keys()
    ])

utils.matplotlib_default()
plt.figure(figsize=(3, 2))
points_x = []
points_y = []

for prop in tqdm.tqdm(utils.PROPS):
    points_x.append(prop)

    points_y_local = []
    # repeat each sampling 10 times to smooth it out
    for _ in range(1):
        data_old_local = copy.deepcopy(data_old)
        data_new = random.sample(data_old_local, k=10)
        data_new_set_i = {x["i"] for x in data_new}
        data_old_local = [x for x in data_old_local if x["i"] not in data_new_set_i]

        while len(data_new) < int(len(data_old)*prop):
            # this is for purely active learning
            # cur_ord = utils.get_sys_ordering(data_new, metric="score")
            # this is true apriori subset selection
            cur_ord = utils.get_sys_ordering(data_new, metric="COMET")

            # TODO: min doesn't make sense here, right?
            # TODO: try both min and max
            line_lowest_conf = min(data_old_local, key=lambda x: ord_distance(cur_ord, x["ord"]))

            # filter that one from the pool (TODO: what if we skip it and keep it there, if it's still "predicted" incorrectly?)
            data_new_set_i.add(line_lowest_conf["i"])
            data_old_local = [x for x in data_old_local if x["i"] not in data_new_set_i]

            data_new.append(line_lowest_conf)


        points_y_local.append(utils.eval_data_pairs(data_new, data_old))

    points_y.append(np.average(points_y_local))

print(f"Average  {np.average(points_y):.2%}")


plt.scatter(
    points_x, points_y,
    marker="o", s=10, color="black",
)
plt.ylabel("Sys. rank accuracy" + " "*5, labelpad=-5)
plt.xlabel("Proportion of original data", labelpad=-2)

ax = plt.gca()
ax.spines[['top', 'right']].set_visible(False)
ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda y, _: '{:.0%}'.format(y))) 
ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda y, _: '{:.0%}'.format(y))) 

plt.ylim(0.7, 1)
plt.tight_layout(pad=0.1)
plt.savefig("figures-v2/active_learning.png", dpi=200)
plt.savefig("figures-v2/active_learning.pdf")
plt.show()