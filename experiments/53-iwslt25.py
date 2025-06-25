# %%
import pickle
import subset2evaluate.utils
import subset2evaluate.evaluate
import subset2evaluate.select_subset
import importlib
importlib.reload(subset2evaluate.utils)

data_all = subset2evaluate.utils.load_data_iwslt()
# with open("../computed/iwslt25_comet.pkl", "rb") as f:
#     data_all = pickle.load(f)

# %%

# inconsistent number of submissions per line
for line in data_all[("iwslt25", "en-de")]:
    print(len(line["scores"]))

# %%

data = data_all[("iwslt25", "en-zh")]
model2score = subset2evaluate.evaluate.get_model_absolute(data)
for cluster in subset2evaluate.evaluate.compute_clusters(data):
    for model in cluster:
        print(f"{model:>50} {model2score[model]:>.2f}")
    print("-"*50)

print("\n")

data = data_all[("iwslt25", "en-ar")]
model2score = subset2evaluate.evaluate.get_model_absolute(data)
for cluster in subset2evaluate.evaluate.compute_clusters(data):
    for model in cluster:
        print(f"{model:>50} {model2score[model]:>.2f}")
    print("-"*50)

print("\n")

data = data_all[("iwslt25", "en-de")]
model2score = subset2evaluate.evaluate.get_model_absolute(data)
for cluster in subset2evaluate.evaluate.compute_clusters(data):
    for model in cluster:
        print(f"{model:>50} {model2score[model]:>.2f}")
    print("-"*50)

# %%
import numpy as np
PROPS = np.linspace(0.05, 1, 10)

data_old = data_all[("iwslt25", "en-zh")]
spa_random_all = []
clu_random_all = []
cor_random_all = []

for _ in range(100):
    data_new_random = subset2evaluate.select_subset.basic(data_old, "random")
    spa_random = subset2evaluate.evaluate.eval_spa(data_new_random, data_old, props=PROPS)
    clu_random, cor_random = subset2evaluate.evaluate.eval_clucor(data_new_random, data_old, props=PROPS)
    spa_random_all.append(spa_random)
    clu_random_all.append(clu_random)
    cor_random_all.append(cor_random)

print(np.average(spa_random_all))

# %%
data_new_metricavg = subset2evaluate.select_subset.basic(data_old, "metric_avg", metric="Unbabel/wmt22-cometkiwi-da")
data_new_metricvar = subset2evaluate.select_subset.basic(data_old, "metric_var", metric="Unbabel/wmt22-cometkiwi-da")
data_new_metriccons = subset2evaluate.select_subset.basic(data_old, "metric_cons", metric="Unbabel/wmt22-cometkiwi-da")
# data_new_metricavg = subset2evaluate.select_subset.basic(data_old, "metric_avg", metric="zouharvi/COMET-partial")
# data_new_metricvar = subset2evaluate.select_subset.basic(data_old, "metric_var", metric="zouharvi/COMET-partial")
# data_new_metriccons = subset2evaluate.select_subset.basic(data_old, "metric_cons", metric="zouharvi/COMET-partial")
spa_metricavg = subset2evaluate.evaluate.eval_spa(data_new_metricavg, data_old, props=PROPS)
spa_metricvar = subset2evaluate.evaluate.eval_spa(data_new_metricvar, data_old, props=PROPS)
spa_metriccons = subset2evaluate.evaluate.eval_spa(data_new_metriccons, data_old, props=PROPS)
clu_metricavg, cor_metricavg = subset2evaluate.evaluate.eval_clucor(data_new_metricavg, data_old, props=PROPS)
clu_metricvar, cor_metricvar = subset2evaluate.evaluate.eval_clucor(data_new_metricvar, data_old, props=PROPS)
clu_metriccons, cor_metriccons = subset2evaluate.evaluate.eval_clucor(data_new_metriccons, data_old, props=PROPS)
print(np.average(spa_metricavg))
print(np.average(spa_metricvar))
print(np.average(spa_metriccons))

# %%

# data_new_diversity_lm = subset2evaluate.select_subset.basic(data_old, "diversity", metric="lm")
data_new_diversity_chrf = subset2evaluate.select_subset.basic(data_old, "diversity", metric="chrf")
# data_new_diversity_unigram = subset2evaluate.select_subset.basic(data_old, "diversity", metric="unigram")
# data_new_diversity_bleu = subset2evaluate.select_subset.basic(data_old, "diversity", metric="bleu")
data_new_diversity = data_new_diversity_chrf
spa_diversity = subset2evaluate.evaluate.eval_spa(data_new_diversity, data_old, props=PROPS)
clu_diversity, cor_diversity = subset2evaluate.evaluate.eval_clucor(data_new_diversity, data_old, props=PROPS)


# %%
import tqdm

spa_kmeans = []
clu_kmeans = []
cor_kmeans = []
load_model = None
for prop in tqdm.tqdm(PROPS):
    k = int(prop * len(data_old))
    data_new_kmeans, load_model = subset2evaluate.select_subset.basic(
        data_old, budget=k, return_model=True, load_model=load_model,
        method="kmeans", features="src",
    )
    spa_kmeans.append(subset2evaluate.evaluate.eval_subset_spa(data_new_kmeans[:k], data_old))
    clu_kmeans.append(subset2evaluate.evaluate.eval_subset_clusters(data_new_kmeans[:k]))
    cor_kmeans.append(subset2evaluate.evaluate.eval_subset_correlation(data_new_kmeans[:k], data_old))

# %%
import matplotlib.pyplot as plt
import utils_fig

fig, axs = plt.subplots(1, 3, sharex=True, figsize=(9, 2.5))
# set default colors
METHODS = [
    {"label": "Random", "color": "black", "data_spa": np.average(spa_random_all, axis=0), "data_cor": np.average(cor_random_all, axis=0), "data_clu": np.average(clu_random_all, axis=0)},
    {"label": "Diversity", "color": utils_fig.COLORS[3], "data_spa": spa_diversity, "data_cor": cor_diversity, "data_clu": clu_diversity},
    {"label": "K-means", "color": utils_fig.COLORS[4], "data_spa": spa_kmeans, "data_cor": cor_kmeans, "data_clu": clu_kmeans},
    {"label": "Metric avg.", "color": utils_fig.COLORS[0], "data_spa": spa_metricavg, "data_cor": cor_metricavg, "data_clu": clu_metricavg},
    {"label": "Metric var.", "color": utils_fig.COLORS[1], "data_spa": spa_metricvar, "data_cor": cor_metricvar, "data_clu": clu_metricvar},
    {"label": "Metric cons.", "color": utils_fig.COLORS[2], "data_spa": spa_metriccons, "data_cor": cor_metriccons, "data_clu": clu_metriccons},
]

for method_i, method_dict in enumerate(METHODS):
    axs[0].plot(
        PROPS,
        method_dict["data_spa"],
        color=method_dict["color"],
    )
    axs[1].plot(
        PROPS,
        method_dict["data_cor"],
        color=method_dict["color"],
        label=method_dict["label"] if method_i < 3 else None,
    )
    axs[2].plot(
        PROPS,
        method_dict["data_clu"],
        color=method_dict["color"],
        label=method_dict["label"] if method_i >= 3 else None,
    )

for ax, data in zip(axs, [spa_random_all, cor_random_all, clu_random_all]):
    random_int = [
        subset2evaluate.utils.confidence_interval(l, confidence=0.9)
        for l in np.array(data).T
    ]
    random_int[-1] = [np.average(np.array(data).T[-1])]*2
    ax.fill_between(
        PROPS,
        [ci[0] for ci in random_int],
        [ci[1] for ci in random_int],
        color="black",
        alpha=0.2,
        linewidth=0,
        label="Random 90% CI" if ax == axs[0] else None,
    )

axs[0].set_ylabel("Soft Pairwise Accuracy")
axs[1].set_ylabel("Kendall $\\tau_b$")
axs[2].set_ylabel("Cluster count")

for ax in axs:
    ax.set_xlabel("Subset size")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xticks([0.05, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["5%", "25%", "50%", "75%", "100%"])


axs[0].legend(
    fontsize=8,
    frameon=False,
    ncol=1,
    loc="lower right",
)
axs[1].legend(
    fontsize=8,
    frameon=False,
    ncol=1,
)
axs[2].legend(
    fontsize=8,
    frameon=False,
    ncol=1,
)

plt.tight_layout(pad=1)
plt.savefig("../figures_pdf/subset2evaluate_random_diversity_kmeans.pdf")
plt.show()

# %%

print({line["doc"] for line in data_all[("iwslt25", "en-zh")]})

import collections