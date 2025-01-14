import subset2evaluate.utils as utils
import utils_fig
import random
import numpy as np
import tqdm
import scipy.stats as st
import matplotlib.pyplot as plt
import subset2evaluate.evaluate

data_old = utils.load_data_wmt()


def confidence_interval(data):
    return st.t.interval(
        confidence=0.95,
        df=len(data) - 1,
        loc=np.mean(data),
        scale=np.std(data)
    )


points_x = []
points_y = []
mean_true = subset2evaluate.evaluate.get_sys_absolute(data_old)

_random = random.Random(0)

for prop in tqdm.tqdm(utils.PROPS):
    k = int(len(data_old) * prop)
    points_x.append(k)

    points_y_local = []
    # repeat each sampling 10 times to smooth it out
    for _ in range(10):
        data_new = _random.sample(data_old, k=k)
        points_y_local.append(np.average(
            [
                abs(mean_true[sys] - sys_mean)
                for sys, sys_mean in subset2evaluate.evaluate.get_sys_absolute(data_new).items()
            ]
        ))

    points_y.append(np.average(points_y_local))


utils_fig.matplotlib_default()
plt.figure(figsize=(3, 2))


plt.plot(
    points_x,
    points_y,
    marker=".",
    markersize=10,
    color=utils_fig.COLORS[0],
    clip_on=False,
    alpha=1,
    linewidth=3,
    label="Estimation error"
)

plt.plot(
    points_x,
    [50 / (x**0.75) for x in points_x],
    marker=".",
    markersize=10,
    color=utils_fig.COLORS[1],
    clip_on=False,
    alpha=1,
    linewidth=3,
    label=r"$50 n^{-3/4}$"
)

# plt.ylabel("Abs difference from true mean", labelpad=-1)
plt.xlabel("Sentences subset size", labelpad=-1)


plt.legend(
    loc="upper right",
    handletextpad=0.3,
    handlelength=1,
    labelspacing=0.2,
    facecolor="#ccc",
    fancybox=False,
    edgecolor=None,
    borderpad=0.1,
    fontsize=9
)

ax = plt.gca()
ax.spines[['top', 'right']].set_visible(False)

# plt.ylim(72, 86)
plt.tight_layout(pad=0.2)
plt.savefig("figures/08a-random_convergence_speed.svg")
plt.show()
