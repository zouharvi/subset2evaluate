import numpy as np
import irt_mt_dev.utils.fig
import matplotlib.pyplot as plt
import matplotlib as mpl
import json

data_irt = json.load(open("computed/irt_MetricX-23-c_noconst_3pl.json", "r"))

irt_mt_dev.utils.fig.matplotlib_default()
plt.figure(figsize=(3, 3))

IS_3PL = "c" in data_irt["items"][0]

if IS_3PL:
    c_min = min([item["c"] for item in data_irt["items"]])
    c_max = max([item["c"] for item in data_irt["items"]])

    cmap = mpl.colormaps.get_cmap("coolwarm_r")
    cnorm = mpl.colors.Normalize(vmin=c_min, vmax=c_max)
    cbar = mpl.cm.ScalarMappable(cmap=cmap, norm=cnorm)


plt.scatter(
    [item["b"] for item in data_irt["items"]],
    [item["a"] for item in data_irt["items"]],
    color="black" if not IS_3PL else [cmap(cnorm(item["c"])) for item in data_irt["items"]],
    s=7,
)

plt.xlabel("Item difficulty", labelpad=0)
plt.ylabel("Item discriminability")
if IS_3PL:
    plt.colorbar(
        cbar,
        ax=plt.gca(),
        orientation="horizontal",
        ticks=[c_min, c_max],
        format="%.2f",
        pad=0.2,
    ).set_label("Item feasability", labelpad=-10)


    print(np.average([item["a"]>=0 for item in data_irt["items"]]))
plt.tight_layout(pad=1)
plt.show()