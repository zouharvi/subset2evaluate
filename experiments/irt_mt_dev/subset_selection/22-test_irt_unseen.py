# %%
import tqdm
import irt_mt_dev.utils as utils
import subset2evaluate.evaluate
import subset2evaluate.select_subset
import numpy as np
import os

os.chdir("/home/vilda/irt-mt-dev")

# %%
data_old_all = list(utils.load_data_wmt_all(normalize=True).items())[:9]
data_train = {}

for data_name, data_old in tqdm.tqdm(data_old_all):
    _data, params = subset2evaluate.select_subset.run_select_subset(data_old, return_model=True, method="pyirt_diffdisc", model="4pl_score", metric="MetricX-23-c", epochs=1000, retry_on_error=True, enforce_positive_disc=True)
    data_train[data_name] = [
        {**line, "irt": line_irt}
        for line, line_irt in zip(data_old, params["items"])
    ]
    print(
        np.average([line["disc"] for line in params["items"]]),
        np.average([line["diff"] for line in params["items"]]),
    )

    # TODO: train multiple times and average item parameters


# %%
data_old = utils.load_data_wmt(year="wmt23", langs="en-cs", normalize=True)
data_new = subset2evaluate.select_subset.run_select_subset(data_old, method="pyirt_unseen_diffdisc", data_train=data_train)
clu_new, acc_new = subset2evaluate.evaluate.run_evaluate_topk(data_old, data_new, metric="human")
print(np.average(clu_new))
print(np.average(acc_new))


# %%
# %%
data_new = subset2evaluate.select_subset.run_select_subset(data_old, method="random", data_train=data_train)
clu_new, acc_new = subset2evaluate.evaluate.run_evaluate_topk(data_old, data_new, metric="human")
print(np.average(clu_new))
print(np.average(acc_new))