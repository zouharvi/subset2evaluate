# %%

import subset2evaluate
import subset2evaluate.evaluate
import subset2evaluate.select_subset
import subset2evaluate.utils
import tqdm
import numpy as np

data_old_all = list(subset2evaluate.utils.load_data("wmt/all").values())[:9]
load_model = None

clu_all = []
cor_all = []
for data_old in tqdm.tqdm(data_old_all):
    for prop in subset2evaluate.utils.PROPS:
        B = int(len(data_old) * prop)
        data_new, load_model = subset2evaluate.select_subset.basic(data_old, method="kmeans", budget=B, load_model=load_model, return_model=True)
        data_new = data_new[:B]
        cor_new = subset2evaluate.evaluate.eval_subset_correlation(data_new, data_old, metric="human")
        clu_new = subset2evaluate.evaluate.eval_subset_clusters(data_new, metric="human")

        clu_all.append(clu_new)
        cor_all.append(cor_new)

print(f"COR: {np.average(cor_all):.1%} | CLU: {np.average(clu_all):.2f}")