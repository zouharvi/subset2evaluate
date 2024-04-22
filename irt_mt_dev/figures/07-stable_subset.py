import utils
import utils.fig
import random
import numpy as np
import tqdm
import collections

data_old = utils.load_data()

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

utils.fig.plot_subsetacc([(points_x, points_y, f"{np.average(points_y):.2%}")], "stable_subset")