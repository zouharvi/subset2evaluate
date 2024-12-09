# %%

import scipy.stats
import irt_mt_dev.utils as utils
import irt_mt_dev.utils.fig
import json
import numpy as np
import matplotlib.pyplot as plt
import subset2evaluate.select_subset
import os

os.chdir("/home/vilda/irt-mt-dev/")

irt_mt_dev.utils.fig.matplotlib_default()

data_irt_all = []

data_all = list(utils.load_data_wmt_all(normalize=True).values())[:9]
for data_old in data_all:
    _data, data_irt = subset2evaluate.select_subset.run_select_subset(
        data_old, method="pyirt_fic", metric="MetricX-23", irt_model="4pl_score", epochs=1000,
        return_model=True
    )
    data_irt_all.append(data_irt)


# %%
data_x_all = [
    [len(item["src"].split()) for item in data_old]
    for data_old in data_all
]
def utility_metricx_avg(item):
    return -np.average(
        [sys_v["MetricX-23"] for sys_v in item["scores"].values()]
    )

def utility_metricx_var(item):
    return np.var(
        [sys_v["MetricX-23"] for sys_v in item["scores"].values()]
    )

def utility_irt_fic(item, data_irt):
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

def utility_irt_diff(item, data_irt):
    item = data_irt["items"][item["i"]]

    # alternatives
    # return item["disc"]
    # return item["feas"]
    return -item["diff"]

data_y_all_metricx_avg = [
    [utility_metricx_avg(item) for item in data_old]    
    for data_old in data_all
]
data_y_all_metricx_var = [
    [utility_metricx_var(item) for item in data_old]
    for data_old in data_all
]
data_y_all_irt_fic = [
    [utility_irt_fic(item, data_irt) for item in data_old]
    for data_old, data_irt in zip(data_all, data_irt_all)
]
data_y_all_irt_diff = [
    [utility_irt_diff(item, data_irt) for item in data_old]
    for data_old, data_irt in zip(data_all, data_irt_all)
]

def z_normalize(data):
    data = np.array(data)
    return (data - np.mean(data)) / np.std(data)

_, axs = plt.subplots(2, 2, figsize=(4, 3))

def plot(ax, title, data_x_all, data_y_all):
    # average pearson correlation
    corr = np.average([
        scipy.stats.pearsonr(data_x, data_y).correlation
        for data_x, data_y in zip(data_x_all, data_y_all)
    ])
    data_x_all = [
        z_normalize(data_x)
        for data_x in data_x_all
    ]
    data_y_all = [
        z_normalize(data_y)
        for data_y in data_y_all
    ]

    # wmt23/de-en
    data_x = data_x_all[2]
    data_y = data_y_all[2]

    if "Information Content" in title:
        # ax.set_yscale("log")
        ax.set_ylim(
            np.quantile(data_y, [0, 0.9]),
        )

    ax.scatter(
        data_x,
        data_y,
        alpha=0.2,
        linewidth=0,
        s=10,
        color="black"
    )
    # corr = np.corrcoef(data_x, data_y)[0, 1]
    ax.set_title(
        title,
        fontsize=9,
        pad=-10,
    )


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

plot(axs[0, 0], "MetricX-23 avg", data_x_all, data_y_all_metricx_avg)
plot(axs[0, 1], "MetricX-23 var", data_x_all, data_y_all_metricx_var)
plot(axs[1, 0], "IRT Information Content", data_x_all, data_y_all_irt_fic)
plot(axs[1, 1], "IRT Difficulty", data_x_all, data_y_all_irt_diff)

axs[0, 0].set_ylabel("Utility")
axs[1, 0].set_ylabel("Utility")
axs[1, 0].set_xlabel("Annotation cost")
axs[1, 1].set_xlabel("Annotation cost")
plt.tight_layout()
plt.savefig("figures_svg/08-subset_cost.svg")
plt.savefig("figures_pdf/08-subset_cost.pdf")
plt.show()