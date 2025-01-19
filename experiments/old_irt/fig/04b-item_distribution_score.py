"""
Look at individual items and how they correspond to model averages.
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
models = list(data_wmt[0]["scores"].keys())

utils_fig.matplotlib_default()

model_scores = {
    model:
    np.average([
        line["scores"][model]["MetricX-23-c"]
        for line in data_wmt
    ])
    for model in models
}
data_x = list(model_scores.values())
data_x_ticks = np.linspace(min(data_x), max(data_x), 100)

model_index = models
model_index.sort(key=lambda model: model_scores[model], reverse=True)
model_index = {
    model: model_index.index(model) + 1
    for model in models
}
print(model_index)
print(model_scores)

fig, axs = plt.subplots(2, 2, figsize=(5, 5))

for ax, item_i in zip(axs.flatten(), [40, 50, 60, 70]):
    ax.set_title(f"Item {item_i}")

    data_y = [data_wmt[item_i]["scores"][model]["MetricX-23-c"] for model in models]

    p, _ = curve_fit(linear, data_x, data_y, maxfev=50000)
    ax.plot(data_x_ticks, linear(data_x_ticks, *p))

    for model in models:
        ax.scatter(
            [model_scores[model]],
            [data_wmt[item_i]["scores"][model]["MetricX-23-c"]],
            color="#ccc",
            linewidth=1,
            edgecolor="black",
            marker="o",
            s=150,
            zorder=10,
        )
        ax.text(
            model_scores[model],
            data_wmt[item_i]["scores"][model]["MetricX-23-c"],
            model_index[model],
            fontsize=7,
            ha="center", va="center",
            zorder=10,
        )
    ax.set_ylabel("Item score")
    ax.set_xlabel("Model average")

plt.tight_layout()

plt.show()
