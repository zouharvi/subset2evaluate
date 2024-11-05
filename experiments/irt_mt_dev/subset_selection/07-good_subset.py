import irt_mt_dev.utils as utils
import irt_mt_dev.utils.fig
import random
import numpy as np
import tqdm
import collections
import multiprocessing

data_old = utils.load_data_wmt()

points_x = []
points_y = []

def run_simulation(k):
    data_new = random.sample(data_old, k=k)
    clusters = utils.eval_system_clusters(data_new, metric="MetricX-23")
    return data_new, clusters

for prop in tqdm.tqdm(utils.PROPS):
    points_x.append(prop)
    k = int(len(data_old)*prop)

    with multiprocessing.Pool(16) as pool:
        results = pool.map(run_simulation, [k]*2000)

    # take best clustering but evaluate on human data
    data_best = max(results, key=lambda x: x[1])
    points_y.append(utils.eval_system_clusters(data_best[0]))
    
print(f"Average  {np.average(points_y):.2f}")

irt_mt_dev.utils.fig.plot_subset_selection([(points_x, points_y, f"{np.average(points_y):.2f}")], "good_subset")