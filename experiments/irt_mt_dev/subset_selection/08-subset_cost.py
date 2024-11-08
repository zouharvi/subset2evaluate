import irt_mt_dev.utils as utils
import irt_mt_dev.utils.fig
import json
import numpy as np
import matplotlib.pyplot as plt

irt_mt_dev.utils.fig.matplotlib_default()

data_irt = json.load(open("computed/irt_wmt_4pl_s0_pyirt.json", "r"))[-1]
data_old = utils.load_data_wmt()

def utility_metricx_avg(item):
    return -np.average(
        [sys_v["MetricX-23"] for sys_v in item["scores"].values()]
    )

def utility_metricx_var(item):
    return np.var(
        [sys_v["MetricX-23"] for sys_v in item["scores"].values()]
    )


_, axs = plt.subplots(2, 2, figsize=(4, 3))

def utility_fisher_information_content(item):
    # aggregared fisher information content
    item = data_irt["items"][item["i"]]

    information = 0
    for theta in data_irt["systems"].values():
        prob = utils.pred_irt(
            theta,
            item
        )
        information += prob*(1-prob)*(item["disc"]**2)
    return information

def utility_irt(item):
    item = data_irt["items"][item["i"]]

    # alternatives
    # return item["disc"]
    # return item["feas"]
    # return item["diff"]
    return item["diff"]+item["disc"]

def plot(ax, title, fn):
    data = [
        (
            len(item["src"].split()),
            fn(item)
        )
        for item in data_old
    ]
    # sort by cost
    data.sort(key=lambda x: x[0], reverse=True)

    ax.scatter(
        [x[0] for x in data],
        [x[1] for x in data],
        alpha=0.4,
        linewidth=0,
        s=5,
        color="black"
    )
    corr = np.corrcoef([x[0] for x in data], [x[1] for x in data])[0, 1]
    ax.set_title(
        title,
        fontsize=9,
        pad=-10,
    )

    if "Information Content" in title:
        # ax.set_yscale("log")
        ax.set_ylim(-0.05, 1)

    ax.text(
        x=1.0,
        y=1.0,
        s=f"$\\rho{{=}}{corr:.3f}$",
        fontsize=9,
        transform=ax.transAxes,
        ha="right",
        va="top",
    )

    ax.set_xticks([], [])
    ax.set_yticks([], [])

    ax.spines[["top", "right"]].set_visible(False)

plot(axs[0, 0], "MetricX-23 avg", utility_metricx_avg)
plot(axs[0, 1], "MetricX-23 var", utility_metricx_var)
plot(axs[1, 0], "IRT Information Content", utility_fisher_information_content)
plot(axs[1, 1], "IRT Difficulty+Discriminability", utility_irt)

axs[0, 0].set_ylabel("Utility")
axs[1, 0].set_ylabel("Utility")
axs[1, 0].set_xlabel("Annotation cost")
axs[1, 1].set_xlabel("Annotation cost")
plt.tight_layout()
plt.savefig("figures_svg/08-subset_cost.svg")
plt.savefig("figures_pdf/08-subset_cost.pdf")
plt.show()