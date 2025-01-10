# %% 
import numpy as np
import pyro
import pyro.distributions
import matplotlib.pyplot as plt
import torch
from matplotlib.lines import Line2D
import utils_fig as figutils

figutils.matplotlib_default()

fig, axs = plt.subplots(1, 2, figsize=(4, 2.5), sharey=True)
axs[0].bar(
    x=np.array([0, 1]),
    height=[0.35, 0.65],
    width=0.15,
    color=figutils.COLORS[1],
    alpha=0.5,
    
)
axs[0].bar(
    x=np.array([1]),
    height=[1],
    width=0.05,
    color=figutils.COLORS[0],
)
axs[0].set_ylabel("Probability / Density")
axs[0].set_xlabel("Binary Succcess")
axs[0].set_xticks(
    [0, 1],
    [" "*5 + "Incorrect", "Correct" + " "*5],
    fontsize=8,
)

axs[0].annotate(
    "$p$=65% =\nIRT prediction",
    xy=(1-0.07, 0.65),
    xytext=(0.15, 0.7),
    arrowprops=dict(
        facecolor='black',
        shrink=0.0,
        width=1,
        headwidth=5,
        headlength=5,
    ),
    fontsize=8,
    ha='left',
    va='center',
)
axs[0].annotate(
    'Observation="correct"',
    xy=(1, 1),
    xytext=(-0.05, 1.25),
    arrowprops=dict(
        facecolor='black',
        shrink=0.0,
        width=1,
        headwidth=5,
        headlength=5,
    ),
    fontsize=8,
    ha='left',
    va='center',
)


from scipy.stats import norm 
  
xs = np.arange(0, 1, 0.01) 
ys = norm.pdf(xs, 0.65, 1/5)
axs[1].fill_between(
    xs, 0, ys,
    alpha=0.5,
    color=figutils.COLORS[1],
    linewidth=0,
)
# axs[1].plot(
#     xs, ys,
#     color=figutils.COLORS[1],
# )
dist_true = [0.8]
axs[1].bar(
    x=np.array(dist_true),
    height=[1],
    width=0.05,
    color=figutils.COLORS[0],
)
axs[1].bar(
    x=[0.65],
    height=[norm.pdf(0.65, 0.65, 1/5)],
    width=0.014,
    color=figutils.COLORS[1],
)
# axs[1].set_ylabel("Prbability / Density")
axs[1].set_xlabel("Continous Succcess")
axs[1].set_xticks(
    [0, 0.65, dist_true[0], 1],
    ["0%", "65%  ", "  80%", "    100%"],
    fontsize=8,
)
axs[1].set_yticks([])
axs[1].set_xlim(-1e-1, 1+1e-1)
                  

axs[1].annotate(
    "$\\mu$=65% =\nIRT prediction",
    xy=(0.65, 0.01),
    xytext=(-0.05, 0.4),
    arrowprops=dict(
        facecolor='black',
        shrink=0.0,
        width=1,
        headwidth=5,
        headlength=5,
    ),
    fontsize=8,
    ha='left',
    va='center',
)
axs[1].annotate(
    "Observation=80%",
    xy=(dist_true[0], 1),
    xytext=(-0.05, 1.25),
    arrowprops=dict(
        facecolor='black',
        shrink=0.0,
        width=1,
        headwidth=5,
        headlength=5,
    ),
    fontsize=8,
    ha='left',
    va='center',
)


axs[0].legend(
    handles=[
        Line2D([0], [0], color=figutils.COLORS[0], lw=5, label='Observation'),
        Line2D([0], [0], color=figutils.COLORS[1], lw=5, label='Predicted'),
    ],
    loc="upper left",
    handletextpad=0.4,
    handlelength=1,
    columnspacing=0.5,
    ncol=2,
)

axs[0].spines[["top", "right"]].set_visible(False)
axs[1].spines[["top", "right"]].set_visible(False)
plt.tight_layout(pad=0.01)
plt.subplots_adjust(wspace=0.1)
plt.savefig("figures_pdf/18-bernoulli_to_normal.pdf")
plt.show()