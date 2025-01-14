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
    cluacc_precomputed_values = pool.starmap(
        subset2evaluate.evaluate.precompute_randnorm,
        [(x[1], 10, "human", 2) for x in data_old_all]
    )
cluacc_precomputed = dict(zip([x[0] for x in data_old_all], cluacc_precomputed_values))

# %%

for method_kwargs in tqdm.tqdm([
    dict(method="metric_var", metric="MetricX-23"),
    dict(method="metric_avg", metric="MetricX-23"),
    dict(method="diversity_bleu"),
    dict(method="pyirt_diffdisc", metric="MetricX-23"),
    dict(method="precomet_diversity"),
    dict(method="precomet_diffdisc"),
]):
    par_clu_all = []
    par_acc_all = []
    for data_name, data_old in data_old_all:
        par_clu, par_acc = subset2evaluate.evaluate.eval_cluacc_randnorm(
            subset2evaluate.select_subset.run_select_subset(data_old, **method_kwargs),
            data_old,
            cluacc_precomputed=cluacc_precomputed[data_name],
        )
        par_clu_all.append(np.average(par_clu))
        par_acc_all.append(np.average(par_acc))
    print(f'{method_kwargs["method"]:<15}', f"CLU: {np.average(par_clu_all):.1%} | ACC: {np.average(par_acc_all):.1%}")