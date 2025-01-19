# %%

import scipy.stats
import subset2evaluate.utils as utils
import utils_fig
import numpy as np
import matplotlib.pyplot as plt
import subset2evaluate.select_subset
import itertools
import sacrebleu

utils_fig.matplotlib_default()

data_irt_all = []

data_all = list(utils.load_data_wmt_all(normalize=True).values())[:9]
for data_old in data_all:
    _data, data_irt = subset2evaluate.select_subset.basic(
        data_old, method="pyirt_diffdisc", metric="MetricX-23-c", model="4pl_score", epochs=1000,
        return_model=True, retry_on_error=True,
    )
    data_irt_all.append(data_irt)


# %%
data_x_all = [
    [len(item["src"].split()) for item in data_old]
    for data_old in data_all
]


def utility_metricx_avg(item):
    return -np.average(
        [model_v["MetricX-23-c"] for model_v in item["scores"].values()]
    )


def utility_metricx_var(item):
    return np.var(
        [model_v["MetricX-23-c"] for model_v in item["scores"].values()]
    )


def utility_irt_diffdisc(item, data_irt):
    item_irt = data_irt["items"][item["i"]]
    return item_irt["diff"] * item_irt["disc"]


metric_bleu = sacrebleu.metrics.BLEU(effective_order=True)


def utility_diversity(line):
    return -np.average([
        metric_bleu.sentence_score(
            text_a,
            [text_b],
        ).score
        for text_a, text_b in itertools.product(line["tgt"].values(), line["tgt"].values())
    ])


data_y_all_metricx_avg = [
    [utility_metricx_avg(item) for item in data_old]
    for data_old in data_all
]
data_y_all_metricx_var = [
    [utility_metricx_var(item) for item in data_old]
    for data_old in data_all
]
data_y_all_irt_diversity = [
    [utility_diversity(item,) for item in data_old]
    for data_old in data_all
]
data_y_all_irt_diffdisc = [
    [utility_irt_diffdisc(item, data_irt) for item in data_old]
    for data_old, data_irt in zip(data_all, data_irt_all)
]


# %%
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
    # flatten
    # data_x = [item for data_x in data_x_all for item in data_x]
    # data_y = [item for data_y in data_y_all for item in data_y]

    if "IRT" in title:
        # ax.set_yscale("log")
        ax.set_ylim(
            np.quantile(data_y, [0.05, 0.95]),
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


plot(axs[0, 0], "MetricX-23 avg.", data_x_all, data_y_all_metricx_avg)
plot(axs[0, 1], "MetricX-23 var.", data_x_all, data_y_all_metricx_var)
plot(axs[1, 0], "Diversity", data_x_all, data_y_all_irt_diversity)
plot(axs[1, 1], "IRT diff.$\\times$disc.", data_x_all, data_y_all_irt_diffdisc)

axs[0, 0].set_ylabel("Utility")
axs[1, 0].set_ylabel("Utility")
axs[1, 0].set_xlabel("Annotation cost")
axs[1, 1].set_xlabel("Annotation cost")
plt.tight_layout()
plt.savefig("figures_svg/11-subset_cost.svg")
plt.savefig("figures_pdf/11-subset_cost.pdf")
plt.show()
