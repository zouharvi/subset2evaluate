"""
Look at individual segments and how they correspond to system averages.
"""

import irt_mt_dev.utils as utils
import irt_mt_dev.utils.fig
import matplotlib.pyplot as plt
import numpy as np

data_wmt = utils.load_data()
systems = list(data_wmt[0]["score"].keys())

irt_mt_dev.utils.fig.matplotlib_default()
plt.figure(figsize=(3, 2))


system_scores = {
    sys:
    np.average([
        line["score"][sys]
        for line in data_wmt
    ])
    for sys in systems
}


plt.scatter(
    [system_scores[sys] for sys in systems],
    [data_wmt[40]["score"][sys] for sys in systems],
    color="black"
)

plt.ylabel("Item score")
plt.xlabel("System average")
plt.tight_layout()

plt.show()