"""
Look at individual segments and how they correspond to system averages.
"""

import json
import irt_mt_dev.utils.fig
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

data = [json.loads(x) for x in open("data/squad_systems.jsonl")]
items = list(data[0]["predictions"].keys())
data = [[sys["predictions"][item]["scores"]["f1"] for item in items] for sys in data]

def linear(x, a, b):
    return a * x + b

irt_mt_dev.utils.fig.matplotlib_default()

data_x = [
    np.average(sys_v)
    for sys_v in data
]
data_x_ticks = np.linspace(min(data_x), max(data_x), 100)

fig, axs = plt.subplots(2, 2, figsize=(5, 5))

for ax, item_i in zip(axs.flatten(), [40, 50, 60, 70]):
    ax.set_title(f"Item {item_i}")
    np.random.seed(0)
    system_subset = np.random.random_integers(0, len(data_x)-1, 15)

    _data_y = np.array([sys_v[item_i] for sys_v in data])[system_subset]
    _data_x = np.array(data_x)[system_subset]

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
    ax.set_xlabel("System average")

plt.tight_layout()
plt.show()