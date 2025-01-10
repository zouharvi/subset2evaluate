import json
import utils
import utils_fig
import matplotlib.pyplot as plt
import numpy as np

utils_fig.matplotlib_default()

item_base = {
    "feas": 0,
    "diff": 0,
    "disc": 1,
}
item_feas = {
    "feas": 0.5,
    "diff": 0,
    "disc": 1,
}
item_disc = {
    "feas": 0,
    "diff": 0,
    "disc": 2,
}
item_diff = {
    "feas": 0,
    "diff": 2,
    "disc": 1,
}
data_x = np.linspace(-2, 4, 100)


_, axs = plt.subplots(ncols=3, figsize=(9.3, 3))

def plot_scenario(ax, item_new, title):
    ax.plot(
        data_x,
        [utils.pred_irt(system_theta, item_base) for system_theta in data_x],
        color="black",
    )
    ax.text(
        x=data_x[0],
        y=utils.pred_irt(data_x[-1], item_base),
        ha="left",
        va="top",
        s="\n".join([f"{k}: {v}" for k, v in item_base.items()]),
    )

    ax.plot(
        data_x,
        [utils.pred_irt(system_theta, item_new) for system_theta in data_x],
        color=utils_fig.COLORS[0],
    )
    ax.text(
        x=data_x[-1],
        y=utils.pred_irt(data_x[-1], item_base)*0.65,
        ha="right",
        va="top",
        s="\n".join([f"{k}: {v}" for k, v in item_new.items()]),
        color=utils_fig.COLORS[0],
    )
    ax.set_xlabel("Student ability")
    ax.set_ylabel("Success")

    ax.set_title(f"Move {title}", fontsize=14)

    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, 1)

plot_scenario(axs[2], item_diff, "difficulty")
plot_scenario(axs[1], item_disc, "discriminability")
plot_scenario(axs[0], item_feas, "feasability")

plt.tight_layout(pad=0)
plt.subplots_adjust(wspace=0.4)
plt.savefig("figures/12b-illustration_move_irt.svg")
plt.show()