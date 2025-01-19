# %%

import subset2evaluate.utils as utils
import numpy as np
import subset2evaluate.evaluate
import subset2evaluate.select_subset
import subset2evaluate.utils
import copy

data_old_all = list(utils.load_data_wmt_all(normalize=True).values())[:9]

data_old = data_old_all[0]
data_old_ord = subset2evaluate.evaluate.get_model_absolute(data_old, metric="MetricX-23")

data_new = copy.deepcopy(data_old)
data_new.sort(
    key=lambda x: subset2evaluate.evaluate.eval_order_accuracy(
        subset2evaluate.evaluate.get_model_absolute([x], metric="MetricX-23"),
        data_old_ord
    ),
    reverse=True
)

clu_new, acc_new = subset2evaluate.evaluate.eval_cluacc(data_new, data_old)
print(f"{np.average(clu_new):.2f} {np.average(acc_new):.1%}")

# TODO: train PreCOMET on this

# %%
clu_new, acc_new = subset2evaluate.evaluate.eval_cluacc(
    subset2evaluate.select_subset.basic(data_old, method="random"),
    data_old,
)
print(f"Random: {np.average(clu_new):.2f} {np.average(acc_new):.1%}")
