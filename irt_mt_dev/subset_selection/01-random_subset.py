import irt_mt_dev.utils as utils
import irt_mt_dev.utils.fig
import random
import numpy as np
import tqdm
import scipy.stats as st

data_old = utils.load_data()

points_x = []
points_y_acc = []
points_y_clu = []
points_y_ci_acc = []
points_y_ci_clu = []

def confidence_interval(data):
    return st.t.interval(
        confidence=0.95,
        df=len(data)-1,
        loc=np.mean(data),
        scale=np.std(data)
    )

for prop in tqdm.tqdm(utils.PROPS):
    points_x.append(prop)

    points_y_local_acc = []
    points_y_local_clu = []
    # repeat each sampling 10 times to smooth it out
    for _ in range(100):
        data_new = random.sample(data_old, k=int(len(data_old)*prop))
        points_y_local_acc.append(utils.eval_order_accuracy(data_new, data_old))
        points_y_local_clu.append(utils.eval_system_clusters(data_new))

    points_y_acc.append(np.average(points_y_local_acc))
    points_y_clu.append(np.average(points_y_local_clu))
    points_y_ci_acc.append(confidence_interval(points_y_local_acc))
    points_y_ci_clu.append(confidence_interval(points_y_local_clu))

print(f"Average ACC {np.average(points_y_acc):.2%}")
print(f"Average CLU {np.average(points_y_clu):.2f}")

irt_mt_dev.utils.fig.plot_subset_selection(
    points=[(points_x, points_y_acc, f"Random {np.average(points_y_acc):.2%}")],
    areas=[(points_x, [x[0] for x in points_y_ci_acc], [x[1] for x in points_y_ci_acc])],
    filename="01-random_subset"
)
irt_mt_dev.utils.fig.plot_subset_selection(
    points=[(points_x, points_y_clu, f"Random {np.average(points_y_clu):.2f}")],
    areas=[(points_x, [x[0] for x in points_y_ci_clu], [x[1] for x in points_y_ci_clu])],
    filename="01-random_subset"
)
