import irt_mt_dev.utils.fig
import matplotlib.pyplot as plt
import matplotlib as mpl
import json

data_irt = json.load(open("computed/irt_MetricX-23-c_noconst_3pl.json", "r"))

irt_mt_dev.utils.fig.matplotlib_default()
plt.figure(figsize=(3, 3))

c_min = min([item["c"] for item in data_irt["items"]])
c_max = max([item["c"] for item in data_irt["items"]])
print(c_min, c_max)

cmap = mpl.colormaps.get_cmap("coolwarm_r")
cnorm = mpl.colors.Normalize(vmin=c_min, vmax=c_max)
cbar = mpl.cm.ScalarMappable(cmap=cmap, norm=cnorm)


plt.scatter(
    [item["b"] for item in data_irt["items"]],
    [item["a"] for item in data_irt["items"]],
    color=[cmap(cnorm(item["c"])) for item in data_irt["items"]],
    s=7,
)

plt.xlabel("Item difficulty")
plt.ylabel("Item discriminability")
plt.colorbar(
    cbar, label="Item fesability",
    orientation="horizontal",
)
plt.tight_layout()

plt.show()