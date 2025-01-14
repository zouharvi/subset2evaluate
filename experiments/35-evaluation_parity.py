# %%

import subset2evaluate.utils as utils
import subset2evaluate.utils
import subset2evaluate.evaluate
import subset2evaluate.select_subset

data_old_all = list(utils.load_data_wmt_all().values())[:9]
data_old = data_old_all[1]


# %%
cluacc_precomputed = subset2evaluate.evaluate.precompute_randnorm(data_old, random_seeds=10, metric="human")

# %%
for method_kwargs in [
    dict(method="metric_var", metric="MetricX-23"),
    dict(method="metric_avg", metric="MetricX-23"),
    dict(method="diversity_bleu"),
    dict(method="pyirt_diffdisc", metric="MetricX-23"),
    dict(method="precomet_diffdisc"),
    dict(method="precomet_diversity"),
]:
    par_clu, par_acc = subset2evaluate.evaluate.run_evaluate_cluacc_randnorm(
        subset2evaluate.select_subset.run_select_subset(data_old, **method_kwargs),
        data_old,
        cluacc_precomputed=cluacc_precomputed,
    )
    print(f'{method_kwargs["method"]:>15}', f"CLU: {par_clu:.1%} | ACC: {par_acc:.1%}")
