import subset2evaluate.utils as utils
import utils_fig
import matplotlib.pyplot as plt
import numpy as np

utils_fig.matplotlib_default()

item_disc = {
    "feas": 0,
    "diff": 0,
    "disc": 2,
}
data_x = np.linspace(-2, 4, 100)


_, axs = plt.subplots(ncols=2, figsize=(6.2, 3))


def plot_scenario(ax, item, systems, title):
    ax.plot(
        data_x,
        [utils.pred_irt(system_theta, item) for system_theta in data_x],
        color=utils_fig.COLORS[0],
    )
    ax.text(
        x=data_x[0],
        y=utils.pred_irt(data_x[-1], item),
        ha="left",
        va="top",
        s="\n".join([f"{k}: {v}" for k, v in item.items()]),
        color=utils_fig.COLORS[0],
    )
    ax.scatter(
        [x for x, _ in systems],
        [utils.pred_irt(x, item) + y for x, y in systems],
        marker="o",
        s=100,
        color="#888",
        label="Individual\nMT systems"
    )
    ax.set_xlabel("System ability")
    ax.set_ylabel("Metric/human score for translation")

    ax.set_title(title, fontsize=14)

    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, 1)


plot_scenario(
    axs[0],
    {
        "feas": 0.2,
        "diff": 0.5,
        "disc": 0.8,
    },
    [
        (2, +0.1),
        (1.6, 0),
        (0, -0.01),
        (0.2, +0.05),
        (-1.3, 0.02),
    ],
    "Source segment 125"
)
plot_scenario(
    axs[1],
    {
        "feas": 0.1,
        "diff": 2,
        "disc": 2,
    },
    [
        (2, +0.05),
        (1.6, 0.3),
        (0, +0.01),
        (0.2, -0.02),
        (-1.3, 0.01),
    ],
    "Source segment 521"
)
axs[0].legend(loc="lower right", handletextpad=0)
plt.tight_layout(pad=0)
plt.subplots_adjust(wspace=0.4)
plt.savefig("figures/12a-illustration_base_irt.svg")
plt.show()
