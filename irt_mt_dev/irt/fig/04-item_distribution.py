import irt_mt_dev.utils.fig
import matplotlib.pyplot as plt
import json

data_irt = json.load(open("computed/irt_MetricX-23-c_0.json", "r"))

irt_mt_dev.utils.fig.matplotlib_default()
plt.figure(figsize=(3, 2))

plt.scatter(
    [item["b"] for item in data_irt["items"]],
    [item["a"] for item in data_irt["items"]],
    color="black",
    s=5,
)

plt.xlabel("Item difficulty")
plt.ylabel("Item discriminability")
plt.tight_layout()

plt.show()