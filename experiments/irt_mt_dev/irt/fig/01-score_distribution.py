import irt_mt_dev.utils as utils
import irt_mt_dev.utils.fig
import matplotlib.pyplot as plt
import random
import numpy as np
import tqdm

data_old = utils.load_data_wmt(normalize=True)
irt_mt_dev.utils.fig.matplotlib_default()
plt.figure(figsize=(3, 2))

systems = list(data_old[0]["scores"].keys())
points_score = [
    sent["scores"][sys]["human"]
    for sent in data_old
    for sys in systems
]
points_metric = [
    sent["scores"][sys]["MetricX-23-c"]
    for sent in data_old
    for sys in systems
]
plt.hist(
    points_score,
    bins=np.linspace(0, 1, 30),
    label="Human", alpha=0.5
)

plt.hist(
    points_metric,
    bins=np.linspace(0, 1, 30),
    label="Metric", alpha=0.5
)

plt.xlabel("Score", labelpad=-10)
plt.xticks([0, 0.25, 0.75, 1.0])
plt.ylabel("Frequency")
plt.yticks([])

irt_mt_dev.utils.fig.turn_off_spines()
plt.tight_layout(pad=0.1)
plt.legend()
plt.show()
