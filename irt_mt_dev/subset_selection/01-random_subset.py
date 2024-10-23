import irt_mt_dev.utils as utils
import irt_mt_dev.utils.fig
import random
import numpy as np
import tqdm

data_old = utils.load_data()

points_x = []
points_y_acc = []
points_y_clu = []

for prop in tqdm.tqdm(utils.PROPS):
    points_x.append(prop)

    points_y_local_acc = []
    points_y_local_clu = []
    # repeat each sampling 10 times to smooth it out
    for _ in range(10):
        data_new = random.sample(data_old, k=int(len(data_old)*prop))
        points_y_local_acc.append(utils.eval_order_accuracy(data_new, data_old))
        points_y_local_clu.append(utils.eval_system_clusters(data_new))
    points_y_acc.append(np.average(points_y_local_acc))
    points_y_clu.append(np.average(points_y_local_clu))

print(f"Average ACC {np.average(points_y_acc):.2%}")
print(f"Average CLU {np.average(points_y_clu):.2f}")

irt_mt_dev.utils.fig.plot_subset_selection(
    [(points_x, points_y_acc, f"Random {np.average(points_y_acc):.2%}")],
    "01-random_subset"
)
irt_mt_dev.utils.fig.plot_subset_selection(
    [(points_x, points_y_clu, f"Random {np.average(points_y_clu):.2f}")],
    "01-random_subset"
)
