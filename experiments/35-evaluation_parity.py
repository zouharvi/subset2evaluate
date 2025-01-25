# %%

import numpy as np
import tqdm
import subset2evaluate.utils as utils
import subset2evaluate.utils
import subset2evaluate.evaluate
import subset2evaluate.select_subset
import multiprocessing

data_old_all = list(utils.load_data_wmt_all().items())[:9]

# %%
with multiprocessing.Pool(len(data_old_all)) as pool:
    clucor_precomputed_values = pool.starmap(
        subset2evaluate.evaluate.precompute_randnorm,
        [(x[1], 10, "human", 2) for x in data_old_all]
    )
clucor_precomputed = dict(zip([x[0] for x in data_old_all], clucor_precomputed_values))

# %%

for method_kwargs in [
    dict(method="metric_avg", metric="MetricX-23"),
    dict(method="metric_var", metric="MetricX-23"),
    dict(method="metric_cons", metric="MetricX-23"),
    dict(method="diversity", metric="BLEU"),
    dict(method="pyirt_diffdisc", metric="MetricX-23"),
    dict(method="cometsrc_avg"),
    dict(method="cometsrc_var"),
    dict(method="local_cometsrc_cons"),
    dict(method="cometsrc_diversity"),
    dict(method="local_cometsrc_diffdisc"),
]:
    par_clu_all = []
    par_cor_all = []
    for data_name, data_old in tqdm.tqdm(data_old_all):
        par_clu, par_cor = subset2evaluate.evaluate.eval_clucor_randnorm(
            subset2evaluate.select_subset.basic(data_old, **method_kwargs),
            data_old,
            clucor_precomputed=clucor_precomputed[data_name],
        )
        par_clu_all.append(np.average(par_clu))
        par_cor_all.append(np.average(par_cor))
    print(f'{method_kwargs["method"]:<15}', f"COR: {np.average(par_cor_all):.1%} | CLU: {np.average(par_clu_all):.1%}")
