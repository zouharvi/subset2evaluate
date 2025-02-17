# %%
import tqdm
import subset2evaluate.utils as utils
import numpy as np
import subset2evaluate.evaluate
import subset2evaluate.select_subset

data_old_all = list(utils.load_data_wmt_all(normalize=True).values())[:9]

points_y_cor = []
points_y_clu = []

for data_old in tqdm.tqdm(data_old_all):
    # the cached data is different for each data_old
    load_model = None
    # run multiple times to smooth variance
    for p in utils.PROPS:
        k = int(len(data_old) * p)
        data_new, load_model = subset2evaluate.select_subset.basic(
            data_old, method="diffuse",
            budget=k,
            return_model=True, load_model=load_model
        )
        data_new = data_new[:k]
        clu_new = subset2evaluate.evaluate.eval_subset_clusters(data_new)
        cor_new = subset2evaluate.evaluate.eval_subset_correlation(data_new, data_old)
        points_y_cor.append(cor_new)
        points_y_clu.append(clu_new)

print(f"COR: {np.average(points_y_cor):.1%} | CLU: {np.average(points_y_clu):.2f}")
