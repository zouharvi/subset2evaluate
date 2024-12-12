
# %%

import irt_mt_dev.utils as utils
import irt_mt_dev.utils.fig
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np
import subset2evaluate.select_subset

import os
os.chdir("/home/vilda/irt-mt-dev/")

def plot_irt(data):
	cmap=cm.coolwarm_r
	norm=mpl.colors.Normalize(
		vmin=min([item["feas"] for item in data["items"]])-0.01,
		vmax=max([item["feas"] for item in data["items"]])+0.01,
	)

	fig, axs = plt.subplots(
		ncols=2, nrows=2,
		width_ratios=(4, 1),
		height_ratios=(1, 4),
	)

	# main plot
	axs[1, 0].scatter(
		[item["diff"] for item in data["items"]],
		[item["disc"] for item in data["items"]],
		s=10,
		alpha=0.5,
		linewidths=0,
		color=[cmap(norm(item["feas"])) for item in data["items"]],
	)
	axs[1, 0].set_ylabel(r"Discriminability ($\alpha$)")
	axs[1, 0].set_xlabel(r"Difficulty ($\beta$)")

	# top histogram (difficulty)
	axs[0, 0].hist(
		[item["diff"] for item in data["items"]],
		bins=np.linspace(*axs[1, 0].get_xlim(), 40),
		orientation="vertical",
		color="black",
	)
	axs[0, 0].set_yticks([])

	# right histogram (discriminability)
	axs[1, 1].hist(
		[item["disc"] for item in data["items"]],
		bins=np.linspace(*axs[1, 0].get_ylim(), 40),
		orientation="horizontal",
		color="black",
	)
	axs[1, 1].set_xticks([])

	# colorbar styling
	fig.colorbar(
		mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
		orientation="horizontal",
		label="Feasability",
		ax=axs[0, 1],
		fraction=1.0,
		aspect=2,
		ticks=[0, 1],
	)
	axs[0, 1].axis("off")

	# plt.gcf().text(0.02, 0.9, f"epoch\n{params_i*100}", fontsize=11)

	pos_theta_tick = axs[1, 0].get_ylim()[0]+(axs[1, 0].get_ylim()[1]-axs[1, 0].get_ylim()[0])*0.1
	axs[1, 0].plot(
		data["systems"].values(),
		len(list(data["systems"].values()))*[pos_theta_tick],
		marker="|",
		alpha=0.5,
		color="black",
	)
      
	plt.tight_layout()


# %%

data_old = list(utils.load_data_wmt_all(normalize=True).values())[0]
_, data_irt_score = subset2evaluate.select_subset.run_select_subset(
    data_old, method="pyirt_fic", metric="MetricX-23-c", irt_model="4pl_score", epochs=1000,
    return_model=True, retry_on_error=True,
)
# _, data_irt_bin = subset2evaluate.select_subset.run_select_subset(
#     data_old, method="pyirt_fic", metric="MetricX-23", irt_model="4pl", epochs=1000,
#     return_model=True
# )

# %%
plot_irt(data_irt_score)
plot_irt(data_irt_bin)