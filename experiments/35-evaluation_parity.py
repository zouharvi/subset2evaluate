# %%


import tqdm
import subset2evaluate.utils as utils
import subset2evaluate.utils
import utils_fig
import numpy as np
import subset2evaluate.evaluate
import subset2evaluate.select_subset

data_old_all = list(utils.load_data_wmt_all().values())[:9]
data_old = data_old_all[1]

acc_random = []
for seed in range(10):
    _, acc_new = subset2evaluate.evaluate.run_evaluate_cluacc(
        subset2evaluate.select_subset.run_select_subset(data_old, method="random", seed=seed),
        data_old,
        metric="human"
    )
    acc_random.append(acc_new)

acc_random = np.average(acc_random, axis=0)

# %%
data_new = subset2evaluate.select_subset.run_select_subset(data_old, method="pyirt_diffdisc", metric="MetricX-23")
par_new = []
for q, acc_random_q in tqdm.tqdm(zip(utils.PROPS, acc_random)):
    acc_new = None
    
    for k in tqdm.tqdm(range(1, len(data_old)+1)):
        acc_metric_var = subset2evaluate.utils.eval_subset_accuracy(data_old, data_new[:k], metric="human")
        if acc_metric_var >= acc_random_q:
            break
    par_new.append(k/(len(data_old)*q))

print(f"{np.average(par_new):.1%}")