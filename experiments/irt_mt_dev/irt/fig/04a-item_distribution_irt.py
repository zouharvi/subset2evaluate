import numpy as np
import irt_mt_dev.utils.fig
import matplotlib.pyplot as plt
import matplotlib as mpl
import json

data_irt = json.load(open("computed/irt_wmt_4pl_s4_eall_metricx.json", "r"))[20]

irt_mt_dev.utils.fig.matplotlib_default()
plt.figure(figsize=(3, 3))

IS_3PL = "feas" in data_irt

if IS_3PL:
    cmap = mpl.colormaps.get_cmap("coolwarm_r")
    cnorm = mpl.colors.Normalize(vmin=0, vmax=1)
    cbar = mpl.cm.ScalarMappable(cmap=cmap, norm=cnorm)


plt.scatter(
    data_irt["diff"],
    data_irt["disc"],
    color="black" if not IS_3PL else [cmap(cnorm(disc)) for disc in data_irt["feas"]],
    s=7,
)

plt.xlabel("Item difficulty", labelpad=0)
plt.ylabel("Item discriminability")
if IS_3PL:
    plt.colorbar(
        cbar,
        ax=plt.gca(),
        orientation="horizontal",
        ticks=[0, 1],
        format="%.2f",
        pad=0.2,
    ).set_label("Item feasability", labelpad=-10)

plt.tight_layout(pad=1)
plt.show()