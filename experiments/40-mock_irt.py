# %%

import matplotlib.pyplot as plt
import utils_fig
import numpy as np

utils_fig.matplotlib_default()


def pred_irt(theta, item):
    return item["feas"] / (1 + np.exp(-item["disc"] * (theta - item["diff"])))


points_x = np.linspace(-3, 5.0, 100)
systems = [-1, 4]

fig, axs = plt.subplots(2, 2, figsize=(8, 4))


def plot_item(ax, item, systems=systems):
    ax.plot(
        points_x,
        [pred_irt(theta, item) for theta in points_x],
        color="black",
        linewidth=2.5,
    )
    ax.scatter(
        systems,
        [pred_irt(theta, item) for theta in systems],
        color=utils_fig.COLORS[0],
        zorder=10,
    )
    ax.text(
        -2.7, 0.55,
        f"$b_s = {item['diff']:.1f}$\n$a_s = {item['disc']:.1f}$\n$c_s = {item['feas']:.1f}$",
        bbox=dict(facecolor='#ddd', edgecolor="white"),
    )


item = {"diff": 0.0, "disc": 1.0, "feas": 1.0}
plot_item(axs[0, 0], item)
axs[0, 0].annotate(
    "IRT predicts low\nscore for a system\nwith low ability",
    xy=(systems[0], pred_irt(systems[0], item)),
    xytext=(1, 0.4),
    arrowprops=dict(arrowstyle="->", color=utils_fig.COLORS[0]),
    fontstyle="italic",
    va="center",
)

item = {"diff": 3.0, "disc": 1.0, "feas": 1.0}
plot_item(axs[0, 1], item)
axs[0, 1].annotate(
    "For difficult ($b_s$) items,\nonly high $\\theta$s lead to\nprediction of success",
    xy=(systems[1], pred_irt(systems[1], item)),
    xytext=(3, 0.8),
    arrowprops=dict(arrowstyle="->", color=utils_fig.COLORS[0]),
    fontstyle="italic",
    va="center",
    ha="right"
)


item = {"diff": 0.0, "disc": 9.0, "feas": 1.0}
plot_item(axs[1, 0], item, systems=[-0.2, 0.5])
axs[1, 0].annotate(
    "Discriminative ($a_s$)\nitems distinguish\nbetween systems of\nclose abilities ($\\theta$)",
    xy=(-0.2, pred_irt(-0.2, item)),
    xytext=(1.3, 0.3),
    arrowprops=dict(arrowstyle="->", color=utils_fig.COLORS[0]),
    fontstyle="italic",
    va="center",
    ha="left"
)
axs[1, 0].annotate(
    "",
    xy=(0.5, pred_irt(0.5, item)),
    xytext=(1.2, 0.3),
    arrowprops=dict(arrowstyle="->", color=utils_fig.COLORS[0]),
)


item = {"diff": 0.0, "disc": 1.0, "feas": 0.8}
plot_item(axs[1, 1], item)
axs[1, 1].annotate(
"Low feasibility ($c_s$)\nprevents any system from\never getting the full score.",
    xy=(systems[1], pred_irt(systems[1], item)),
    xytext=(4.8, 0.2),
    arrowprops=dict(arrowstyle="->", color=utils_fig.COLORS[0]),
    fontstyle="italic",
    va="center",
    ha="right"
)

for ax in axs.flatten():
    ax.set_xlim(-3, 5)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlabel("System ability $\\theta$", labelpad=-1)
    ax.set_ylabel("Predicted success", labelpad=-2)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticklabels(['0%', '25%', '50%', '75%', '100%'])

plt.tight_layout(pad=0.1, h_pad=1, w_pad=0.5)
plt.savefig("../figures_pdf/40-mock_irt.pdf")
plt.savefig("../figures_svg/40-mock_irt.svg")
plt.show()