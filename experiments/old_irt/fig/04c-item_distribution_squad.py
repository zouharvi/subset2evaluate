"""
Look at individual items and how they correspond to model averages.
"""

import utils_fig
import subset2evaluate.utils as utils
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

data = utils.load_data_squad()
models = list(data[0]["scores"].keys())
data = [[item["scores"][model]["f1"] for item in data] for model in models]


def linear(x, a, b):
    return a * x + b


utils_fig.matplotlib_default()

data_x = [
    np.average(model_v)
    for model_v in data
]

fig, axs = plt.subplots(2, 2, figsize=(5, 5))

for ax, item_i in zip(axs.flatten(), [40, 50, 60, 70]):
    ax.set_title(f"Item {item_i}")
    np.random.seed(0)
    model_subset = np.random.random_integers(0, len(data_x) - 1, 15)

    _data_y = np.array([model_v[item_i] for model_v in data])[model_subset]
    _data_x = np.array(data_x)[model_subset]
    data_x_ticks = np.linspace(min(_data_x), max(_data_x), 100)

    p, _ = curve_fit(linear, _data_x, _data_y, maxfev=50000)
    ax.plot(data_x_ticks, linear(data_x_ticks, *p))

    ax.scatter(
        _data_x,
        _data_y,
        color="black",
        alpha=0.4,
        linewidth=0,
        marker="o",
        s=150,
        zorder=10,
    )
    ax.set_ylabel("Item score")
    ax.set_xlabel("Model average")

plt.tight_layout()
plt.show()
