import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from PIL import Image
import json
import cv2
from multiprocessing import Pool

def fig2img(fig):
	"""Convert a Matplotlib figure to a PIL Image and return it"""
	import io
	buf = io.BytesIO()
	fig.savefig(buf, dpi=200)
	buf.seek(0)
	img = Image.open(buf)
	return img

def plot_irt(x):
	params_i, data = x
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

	plt.gcf().text(0.02, 0.9, f"epoch\n{params_i*100}", fontsize=11)

	pos_theta_tick = axs[1, 0].get_ylim()[0]+(axs[1, 0].get_ylim()[1]-axs[1, 0].get_ylim()[0])*0.1
	axs[1, 0].plot(
		data["systems"].values(),
		len(list(data["systems"].values()))*[pos_theta_tick],
		marker="|",
		alpha=0.5,
		color="black",
	)

	plt.tight_layout()
	img = fig2img(fig)
	
	return img

# for seed in [0, 1, 2, 3, 4]:
for seed in [0]:
	# paralelize
	with Pool(10) as pool:
		imgs = pool.map(
			plot_irt,
			# skip zeroth epoch
			enumerate(json.load(open(f"computed/irt_wmt_4pl_s{seed}_pyirt.json"))[1:])
		)

		# compute video
		videodims = (imgs[0].width, imgs[0].height)
		fourcc = cv2.VideoWriter_fourcc(*'mp4v')    
		video = cv2.VideoWriter(f"computed/irt_squad_4pl_s{seed}_pyirt.mp4", fourcc, 4, videodims)
		img = Image.new('RGB', videodims, color = 'darkred')

		for img in imgs:
			# draw frame specific stuff here.
			video.write(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))

		video.release()