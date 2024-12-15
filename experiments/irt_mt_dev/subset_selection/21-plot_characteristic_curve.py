


# %%

import copy
import irt_mt_dev.utils as utils
import irt_mt_dev.utils.fig as figutils
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np
import subset2evaluate.select_subset
import pickle

import os
os.chdir("/home/vilda/irt-mt-dev/")

# %%
data_old = list(utils.load_data_wmt_all(normalize=True).values())[3]
data_old_bin = list(utils.load_data_wmt_all(normalize=True, binarize=True).values())[3]
_, data_irt_score = subset2evaluate.select_subset.run_select_subset(
    data_old, method="pyirt_fic", metric="MetricX-23-c", model="4pl_score", epochs=2000,
    return_model=True, retry_on_error=True,
)
_, data_irt_bin = subset2evaluate.select_subset.run_select_subset(
    data_old, method="pyirt_fic", metric="MetricX-23-c", model="4pl", epochs=1000,
    return_model=True
)

# %%
# save data because computing has noise
with open("computed/21-plot_characteristic_curve.pkl", "wb") as f:
    pickle.dump((data_irt_score, data_irt_bin, data_old, data_old_bin), f)

# %%
# load data again
with open("computed/21-plot_characteristic_curve.pkl", "rb") as f:
    data_irt_score, data_irt_bin, data_old, data_old_bin = pickle.load(f)

# %%

import importlib
import irt_mt_dev.utils
importlib.reload(irt_mt_dev.utils)
figutils.matplotlib_default()

def pred_irt_4pl(theta, item):
    return item["feas"] / (1 + np.exp(-item["disc"] * (theta - item["diff"])))

def plot_item_curve(ax, data_irt, data_old, title, item_i=87):
    data_old_item = data_old[item_i]
    data_irt_item = data_irt["items"][item_i]
    systems = list(data_irt["systems"].keys())
    theta_min = min(list(data_irt["systems"].values()))
    theta_max = max(list(data_irt["systems"].values()))

    
    data_x = np.linspace(theta_min, theta_max, 100)
    data_y = [pred_irt_4pl(theta, data_irt_item) for theta in data_x]
    ax.plot(
        data_x,
        data_y,
        color="black"
    )
    ax.scatter(
        list(data_irt["systems"].values()),
        [data_old_item["scores"][system]["MetricX-23-c"] for system in systems],
        zorder=20,
        color=figutils.COLORS[0] if "Cont" in title else figutils.COLORS[1],
    )
    ax.set_title(title, fontsize=10)
    ax.set_ylim(0-0.1, 1+0.1)
    ax.spines[["top", "right"]].set_visible(False)


fig, axs = plt.subplots(1, 2, figsize=(4, 2))

plot_item_curve(axs[0], data_irt_bin, data_old_bin, "Binary IRT", item_i=128)
plot_item_curve(axs[1], data_irt_score, data_old, "Continuous IRT", item_i=128)

axs[0].set_ylabel("$\\bf Item$ success")
axs[0].set_xlabel(" " * 40 + "System ability ($\\theta$)")
axs[0].set_yticks([0, 1])
axs[0].yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter("{x:.1f}"))
axs[1].yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter("{x:.1f}"))
axs[1].set_xlim(0.22, 0.78)
plt.tight_layout()
plt.savefig("figures_pdf/21-item_characteristic_curve.pdf")
plt.show()

# %%
# find examples with minimum avg loss
y_pred = [(i, [pred_irt_4pl(theta, item) for theta in data_irt_score["systems"].values()]) for i, item in enumerate(data_irt_score["items"])]
y_true = [(i, [data_old[i]["scores"][system]["MetricX-23-c"] for system in data_irt_score["systems"]]) for i in range(len(data_old))]
losses = [(i, np.mean((np.array(y_pred[i][1]) - np.array(y_true[i][1]))**2)) for i in range(len(data_old))]
sorted(losses, key=lambda x: x[1])[:20]
sorted([(x["disc"], i) for i, x in enumerate(data_irt_score["items"])], key=lambda x: x[0], reverse=True)[:20]

# find examples with minimum avg loss
y_pred = [(i, [pred_irt_4pl(theta, item) for theta in data_irt_bin["systems"].values()]) for i, item in enumerate(data_irt_bin["items"])]
y_true = [(i, [data_old_bin[i]["scores"][system]["MetricX-23-c"] for system in data_irt_bin["systems"]]) for i in range(len(data_old_bin))]
losses = [(i, np.mean((np.array(y_pred[i][1]) - np.array(y_true[i][1]))**2)) for i in range(len(data_old_bin))]
losses = [
    (i, l) for i, l in losses if
    np.mean([data_old_bin[i]["scores"][system]["MetricX-23-c"] for system in data_irt_bin["systems"]]) > 0.2 and 
    np.mean([data_old_bin[i]["scores"][system]["MetricX-23-c"] for system in data_irt_bin["systems"]]) < 0.8
]
sorted(losses, key=lambda x: x[1])[100:120]


# %%
# plot test curve


def plot_test_curve(ax, data_irt, data_old, title):
    systems = list(data_irt["systems"].keys())
    theta_min = min(list(data_irt["systems"].values()))
    theta_max = max(list(data_irt["systems"].values()))

    data_x = np.linspace(theta_min, theta_max, 100)
    data_y_true_all = []
    data_y_pred_all = []

    for item_i in range(len(data_old)):
        data_old_item = data_old[item_i]
        data_irt_item = data_irt["items"][item_i]
        data_y_pred_all.append([pred_irt_4pl(theta, data_irt_item) for theta in data_x])
        data_y_true_all.append([data_old_item["scores"][system]["MetricX-23-c"] for system in systems])
    
    ax.plot(
        data_x,
        np.average(data_y_pred_all, axis=0),
        color="black"
    )
    ax.scatter(
        list(data_irt["systems"].values()),
        np.average(data_y_true_all, axis=0),
        zorder=20,
        color=figutils.COLORS[0] if "Cont" in title else figutils.COLORS[1],
    )
    ax.set_title(title, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)

fig, axs = plt.subplots(1, 2, figsize=(4, 2))
plot_test_curve(axs[0], data_irt_bin, data_old_bin, "Binary IRT")
plot_test_curve(axs[1], data_irt_score, data_old, "Continuous IRT")


axs[0].set_ylabel("$\\bf Test$ success")
axs[0].set_xlabel(" " * 40 + "System ability ($\\theta$)")
axs[0].yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter("{x:.1f}"))
axs[1].yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter("{x:.1f}"))
axs[1].set_xlim(0.22, 0.78)
plt.tight_layout()
plt.savefig("figures_pdf/21-test_characteristic_curve.pdf")
plt.show()