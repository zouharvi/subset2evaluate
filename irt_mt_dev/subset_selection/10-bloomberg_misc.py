import irt_mt_dev.utils as utils
import irt_mt_dev.utils.fig
import scipy
import numpy as np
import tqdm
import random
import copy

data = utils.load_data(year="wmt23", langs="cs-uk")

data.sort(key=lambda x: len(x["src"]))
for line in data[:10]:
    print(line["src"])

def heuristic_corr(line):
    return scipy.stats.spearmanr(
        [sys_v["COMET"] for sys_v in line["metrics"].values()],
        [sys_v["MetricX-23-c"] for sys_v in line["metrics"].values()]
    )[0]
def heuristic_abs(line):
    # np.max is also good
    return np.average(
        [sys_v["MetricX-23-c"] for sys_v in line["metrics"].values()]
    )

def heuristic_std(line):
    return np.std([sys_v["MetricX-23"] for sys_v in line["metrics"].values()])

points_x_all = []
points_y_rand_all = []
points_y_var_all = []
points_y_len_all = []

for year, langs in [("wmt23", "en-de"), ("wmt23", "cs-uk"), ("wmt23", "en-cs"), ("wmt23", "en-ja")]:
    points_x = []
    points_y_rand = []
    points_y_var = []
    points_y_len = []
    data_old = utils.load_data(year=year, langs=langs)

    data_old_len = copy.deepcopy(data_old)
    data_old_var = copy.deepcopy(data_old)
    # sort by the heuristic
    data_old_var = [(line, heuristic_corr(line)) for line in tqdm.tqdm(data_old_var)]
    data_old_var.sort(key=lambda x: x[1])
    data_old_var = [x[0] for x in data_old_var]

    data_old_len = [(line, len(line["src"])) for line in tqdm.tqdm(data_old_len)]
    data_old_len.sort(key=lambda x: x[1])
    data_old_len = [x[0] for x in data_old_len]

    for prop in tqdm.tqdm(utils.PROPS):
        points_x.append(prop)

        points_y_var.append(
            utils.eval_data_pairs(data_old_var[-int(len(data_old_var) * prop):], data_old)
        )

        points_y_len.append(
            utils.eval_data_pairs(data_old_len[-int(len(data_old_len) * prop):], data_old)
        )

        points_y_rand_local = []
        for _ in range(10):
            data_local = random.choices(data_old, k=int(len(data_old) * prop))
            points_y_rand_local.append(
                utils.eval_data_pairs(data_local, data_old)
            )
        points_y_rand.append(np.average(points_y_rand_local))

    points_x_all.append(points_x)
    points_y_rand_all.append(points_y_rand)
    points_y_var_all.append(points_y_var)
    points_y_len_all.append(points_y_len)

# average points
points_y_rand = np.average(points_y_rand_all, axis=0)
points_y_var = np.average(points_y_var_all, axis=0)
points_y_len = np.average(points_y_len_all, axis=0)


def plot_subsetacc(points, filename=None):
    import matplotlib.pyplot as plt

    irt_mt_dev.utils.fig.matplotlib_default()
    plt.figure(figsize=(4, 1.1))

    colors = ["#b30000", "#4421af", "black"]

    for (points_x, points_y, zorder, label), color in zip(points, colors):
        # smooth
        points_y = [np.average(points_y[i:i+2]) for i in range(len(points_y))]
        plt.plot(
            points_x,
            points_y,
            marker="o",
            markersize=5,
            color=color,
            label=label,
            clip_on=False if min(points_y) > 0.65 else True,
            linewidth=2,
            zorder=zorder,
        )

    plt.ylabel("Rank.\naccuracy", labelpad=-5)
    plt.xlabel("Proportion of original data", labelpad=-8)

    ax = plt.gca()
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xticks([0.1, 0.9])
    ax.set_xticklabels(["10%", "90%"])
    ax.set_yticks([0.87, 0.98])
    ax.set_yticklabels(["85%", "100%"])

    plt.legend(
        loc="lower right",
        handletextpad=0.4,
        handlelength=1,
        labelspacing=0.2,
        scatteryoffsets=[0.5]*len(points),
        fontsize=8,
    )

    plt.ylim(0.87, 0.98)
    plt.tight_layout(pad=0.0)
    if filename:
        plt.savefig(f"figures/{filename}.pdf")
    plt.show()

plot_subsetacc(
    [
        (utils.PROPS, points_y_var, 10, f"Variance-based {np.average(points_y_var):.2%}"),
        (utils.PROPS, points_y_len, 0, f"Length-based {np.average(points_y_len):.2%}"),
        (utils.PROPS, points_y_rand, 5, f"Random {np.average(points_y_rand):.2%}"),
    ],
    "bloomberg_tmp",
)
