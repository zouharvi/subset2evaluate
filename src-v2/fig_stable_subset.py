import utils
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import tqdm
import collections
import typing

data_old = utils.load_data()

utils.matplotlib_default()
plt.figure(figsize=(3, 2))
points_x = []
points_y = []

for prop in tqdm.tqdm(utils.PROPS):
# for prop in tqdm.tqdm([0.25, 0.5, 0.75]):
    points_x.append(prop)
    k = int(len(data_old)*prop)

    points_y_local = []
    # repeat each sampling 10 times to smooth it out
    for _ in range(10):
        subsets = collections.defaultdict(list)
        for _ in range(10000):
            data_new = random.sample(data_old, k=k)
            ord_new = tuple(utils.get_sys_ordering(data_new, metric="MetricX-23-c").keys())
            subsets[ord_new].append(data_new)
        
        subset_most_stable = max(subsets.items(), key=lambda x: len(x[1]))
        lines_most_stable = collections.Counter()
        for subset in subset_most_stable[1]:
            for line in subset:
                lines_most_stable[line["i"]] += 1

        lines_most_stable = lines_most_stable.most_common(k)
        data_new = [data_old[line_i] for line_i, _ in lines_most_stable]
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
plt.savefig("figures-v2/stable_subset.png", dpi=200)
plt.savefig("figures-v2/stable_subset.pdf")
plt.show()