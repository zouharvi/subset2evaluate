import utils
import random
import numpy as np
import tqdm

data_old = utils.load_data()

points_x = []
points_y = []

for prop in tqdm.tqdm(utils.PROPS):
    points_x.append(prop)

    points_y_local = []
    # repeat each sampling 10 times to smooth it out
    for _ in range(100):
        data_new = random.sample(data_old, k=int(len(data_old)*prop))
        points_y_local.append(utils.eval_data_pairs(data_new, data_old))

    points_y.append(np.average(points_y_local))
    
print(f"Average  {np.average(points_y):.2%}")

utils.plot_single(points_x, points_y, "random_subset")
