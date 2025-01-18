"""
Look at individual items and how they correspond to system averages.
"""

import subset2evaluate.utils as utils
import utils_fig
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


def sigmoid_irt(x, a, b, c):
    return c / (1 + np.exp(-a * (x - b)))


def linear(x, a, b):
    return a * x + b


data_wmt = utils.load_data_wmt(normalize=True)
systems = list(data_wmt[0]["scores"].keys())

utils_fig.matplotlib_default()

system_scores = {
    sys:
    np.average([
        line["scores"][sys]["MetricX-23-c"]
        for line in data_wmt
    ])
    for sys in systems
}
data_x = list(system_scores.values())
data_x_ticks = np.linspace(min(data_x), max(data_x), 100)

system_index = systems
system_index.sort(key=lambda sys: system_scores[sys], reverse=True)
system_index = {
    sys: system_index.index(sys) + 1
    for sys in systems
}
print(system_index)
print(system_scores)

fig, axs = plt.subplots(2, 2, figsize=(5, 5))

for ax, item_i in zip(axs.flatten(), [40, 50, 60, 70]):
    ax.set_title(f"Item {item_i}")

    data_y = [data_wmt[item_i]["scores"][sys]["MetricX-23-c"] for sys in systems]

    p, _ = curve_fit(linear, data_x, data_y, maxfev=50000)
    ax.plot(data_x_ticks, linear(data_x_ticks, *p))

    for sys in systems:
        ax.scatter(
            [system_scores[sys]],
            [data_wmt[item_i]["scores"][sys]["MetricX-23-c"]],
            color="#ccc",
            linewidth=1,
            edgecolor="black",
            marker="o",
            s=150,
            zorder=10,
        )
        ax.text(
            system_scores[sys],
            data_wmt[item_i]["scores"][sys]["MetricX-23-c"],
            system_index[sys],
            fontsize=7,
            ha="center", va="center",
            zorder=10,
        )
    ax.set_ylabel("Item score")
    ax.set_xlabel("System average")

plt.tight_layout()

plt.show()
