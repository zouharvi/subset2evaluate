
# %%

import subset2evaluate.utils as utils
import utils_fig as figutils
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np
import subset2evaluate.select_subset


def plot_irt(data):
    cmap = cm.coolwarm_r
    norm = mpl.colors.Normalize(
        vmin=0 - 0.1,
        vmax=1 + 0.1,
        # vmin=min([item["feas"] for item in data["items"]])-0.01,
        # vmax=max([item["feas"] for item in data["items"]])+0.01,
    )

    fig, axs = plt.subplots(
        ncols=2, nrows=2,
        width_ratios=(1, 7),
        height_ratios=(7, 1),
        figsize=(4, 3),
    )
    sys_mean = np.average(list(data["systems"].values()))

    # main plot
    axs[0, 1].scatter(
        [item["diff"] - sys_mean for item in data["items"]],
        [item["disc"] for item in data["items"]],
        s=5,
        alpha=0.7,
        linewidths=0,
        color=[cmap(norm(item["feas"])) for item in data["items"]],
        color=figutils.COLORS[1],
    )
    axs[0, 1].spines[["bottom", "top", "left", "right"]].set_visible(False)
    axs[0, 1].set_xticks([])
    axs[0, 1].set_yticks([])

    # top histogram (difficulty)
    axs[1, 1].hist(
        [item["diff"] - sys_mean for item in data["items"]],
        bins=np.linspace(*axs[0, 1].get_xlim(), 40),
        orientation="vertical",
        color="black",
    )
    axs[1, 1].set_yticks([])
    axs[1, 1].xaxis.set_ticks_position('top')
    axs[1, 1].invert_yaxis()
    axs[1, 1].set_xlabel(" " * 10 + r"Difficulty ($b$)", labelpad=-10)
    axs[1, 1].spines[["left", "bottom", "right"]].set_visible(False)

    # right histogram (discriminability)
    axs[0, 0].hist(
        [item["disc"] for item in data["items"]],
        bins=np.linspace(*axs[0, 1].get_ylim(), 40),
        orientation="horizontal",
        color="black",
    )
    axs[0, 0].invert_xaxis()
    axs[0, 0].set_xticks([])
    axs[0, 0].yaxis.set_ticks_position('right')
    axs[0, 0].set_ylabel(" " * 15 + r"Discriminability ($a$)", labelpad=-20)
    axs[0, 0].spines[["bottom", "top", "left"]].set_visible(False)

    axs[1, 0].axis("off")

    pos_theta_tick = axs[0, 1].get_ylim()[0] + (axs[0, 1].get_ylim()[1] - axs[0, 1].get_ylim()[0]) * 0.1
    axs[0, 1].plot(
        [5 * (x - sys_mean) for x in data["systems"].values()],
        len(list(data["systems"].values())) * [pos_theta_tick],
        marker=".",
        alpha=1,
        markersize=10,
        # linewidth=0,
        color=figutils.COLORS[0],
    )
    axs[0, 1].text(
        0 - sys_mean * 0.7, pos_theta_tick * 0.4,
        "System\nability ($\\theta$)",
        ha="right",
        va="top",
        color=figutils.COLORS[0],
        fontweight="bold",
        fontsize=9,
    )

    # annotations
    axs[0, 1].annotate(
        "Annotation\nerror",
        xy=(0.7, 0.2),
        xytext=(0.8, 0.5),
        textcoords="axes fraction",
        xycoords="axes fraction",
        ha="center",
        fontweight="bold",
        fontsize=9,
        arrowprops=dict(arrowstyle="->"),
    )
    axs[0, 1].annotate(
        "Easy",
        xy=(0.2, 0.35),
        xytext=(0.2, 0.6),
        textcoords="axes fraction",
        xycoords="axes fraction",
        ha="center",
        fontweight="bold",
        fontsize=9,
        arrowprops=dict(arrowstyle="->"),
    )

    axs[0, 1].annotate(
        "Difficult &\ndiscriminative",
        xy=(0.5, 0.7),
        xytext=(0.8, 0.8),
        textcoords="axes fraction",
        xycoords="axes fraction",
        ha="center",
        fontweight="bold",
        fontsize=9,
        arrowprops=dict(arrowstyle="->"),
    )

    plt.tight_layout(pad=0)
    plt.subplots_adjust()
    plt.savefig("figures_pdf/20-annotated_irt.pdf")
    plt.show()


# %%
data_old = list(utils.load_data_wmt_all(normalize=True).values())[2]

# %%
# renormalize without outliers
# data_old_norm = copy.deepcopy(data_old)
# def renormalize(data_y):
#     data_y = np.array(data_y)
#     min_y, max_y = np.quantile(data_y, [0.01, 0.99])
#     data_y = np.clip(data_y, min_y, max_y)

#     data_y = (data_y - min_y) / (max_y - min_y)
#     mean = np.mean(data_y)

#     return data_y
# data_y = []
# systems = list(data_old_norm[0]["scores"].keys())
# for line in data_old_norm:
#     for sys in systems:
#         data_y.append(line["scores"][sys]["MetricX-23-c"])
# data_y = list(renormalize(data_y))
# for line in data_old_norm:
#     for sys in systems:
#         line["scores"][sys]["MetricX-23-c"] = data_y.pop(0)
#     # offset = 0.5 - mean
#     # return [y+offset for y in data_y]


# %%
_, data_irt_score = subset2evaluate.select_subset.run_select_subset(
    data_old, method="pyirt_fic", metric="MetricX-23-c", model="4pl_score", epochs=2000,
    return_model=True, retry_on_error=True,
)
plot_irt(data_irt_score)
# %%
_, data_irt_bin = subset2evaluate.select_subset.run_select_subset(
    data_old, method="pyirt_fic", metric="MetricX-23", model="4pl", epochs=1000,
    return_model=True
)
plot_irt(data_irt_bin)
