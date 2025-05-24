# %%

import subset2evaluate
import subset2evaluate.evaluate
import subset2evaluate.select_subset
import subset2evaluate.utils
import utils_fig
import tqdm
import numpy as np

data_old_all = list(subset2evaluate.utils.load_data("wmt/all").values())[:9]
load_model = None

# %%
clu_kmeans = []
cor_kmeans = []
for data_old in tqdm.tqdm(data_old_all):
    clu_local = []
    cor_local = []
    for prop in subset2evaluate.utils.PROPS:
        B = int(len(data_old) * prop)
        data_new, load_model = subset2evaluate.select_subset.basic(data_old, method="kmeans", budget=B, load_model=load_model, return_model=True)
        data_new = data_new[:B]
        cor_new = subset2evaluate.evaluate.eval_subset_correlation(data_new, data_old, metric="human")
        clu_new = subset2evaluate.evaluate.eval_subset_clusters(data_new, metric="human")

        clu_local.append(clu_new)
        cor_local.append(cor_new)

    clu_kmeans.append(clu_local)
    cor_kmeans.append(cor_local)

# %%
clu_random = []
cor_random = []
for data_old in tqdm.tqdm(data_old_all):
    for _ in range(100):
        clu_new, cor_new = subset2evaluate.evaluate.eval_clu_cor(
            subset2evaluate.select_subset.basic(data_old, method="random"),
            data_old,
            metric="human"
        )
        clu_random.append(clu_new)
        cor_random.append(cor_new)


# %%
utils_fig.plot_subset_selection(
    points=[
        (subset2evaluate.utils.PROPS, np.average(clu_random, axis=0), f"Random {np.average(clu_random):.2f}"),
        (subset2evaluate.utils.PROPS, np.average(clu_kmeans, axis=0), f"k-means {np.average(clu_kmeans):.2f}"),
    ],
    filename="43-kmeans",
    colors=["#000000"] + utils_fig.COLORS,
    height=1.5,
)

utils_fig.plot_subset_selection(
    points=[
        (subset2evaluate.utils.PROPS, np.average(cor_random, axis=0), f"Random {np.average(cor_random):.1%}"),
        (subset2evaluate.utils.PROPS, np.average(cor_kmeans, axis=0), f"k-means {np.average(cor_kmeans):.1%}"),
    ],
    filename="43-kmeans",
    colors=["#000000"] + utils_fig.COLORS,
    height=1.5,
)
