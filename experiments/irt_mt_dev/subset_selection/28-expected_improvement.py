# %%
import collections
import irt_mt_dev.utils as utils
import subset2evaluate.evaluate
import subset2evaluate.select_subset
import numpy as np
import os
import tqdm
import matplotlib.pyplot as plt

os.chdir("/home/vilda/irt-mt-dev")

data_old_all = list(utils.load_data_wmt_all(normalize=True).values())[:9]

# %%
# aggregate scores

acc_new_all_hum = collections.defaultdict(list)
clu_new_all_hum = collections.defaultdict(list)
acc_new_all_met = collections.defaultdict(list)
clu_new_all_met = collections.defaultdict(list)

for data_old in tqdm.tqdm(data_old_all):
    for method_kwargs in [
        dict(method="random"),
        dict(method="avg"),
        dict(method="var"),
        dict(method="output_text_var"),
        # dict(method="pyirt_diffdisc", model="4pl_score", epochs=1000),
    ]:
        # run multiple times to average out the effect
        acc_new_all_hum_local = []
        clu_new_all_hum_local = []
        acc_new_all_met_local = []
        clu_new_all_met_local = []
        for _ in range(5 if method_kwargs["method"] == "pyirt_diffdisc" else 10 if method_kwargs["method"] == "random" else 1):
            data_new = subset2evaluate.select_subset.run_select_subset(
                data_old,
                **method_kwargs,
                metric="MetricX-23-c",
                retry_on_error=True,
            )
            clu_new, acc_new = subset2evaluate.evaluate.run_evaluate_topk(data_old, data_new, metric="human")
            acc_new_all_hum_local.append(np.average(acc_new))
            clu_new_all_hum_local.append(np.average(clu_new))

            clu_new, acc_new = subset2evaluate.evaluate.run_evaluate_topk(data_old, data_new, metric="MetricX-23-c")
            acc_new_all_met_local.append(np.average(acc_new))
            clu_new_all_met_local.append(np.average(clu_new))

        acc_new_all_hum[method_kwargs["method"]].append(np.average(acc_new_all_hum_local))
        clu_new_all_hum[method_kwargs["method"]].append(np.average(clu_new_all_hum_local))
        acc_new_all_met[method_kwargs["method"]].append(np.average(acc_new_all_met_local))
        clu_new_all_met[method_kwargs["method"]].append(np.average(clu_new_all_met_local))


# %%

plt.scatter(
    clu_new_all_met["random"],
    clu_new_all_hum["random"],
    label="random"
)
plt.scatter(
    clu_new_all_met["var"],
    clu_new_all_hum["var"],
    label="var"
)
plt.legend()
plt.show()

# %%

plt.scatter(
    np.array(clu_new_all_met["var"]) - np.array(clu_new_all_met["random"]),
    np.array(clu_new_all_hum["var"]) - np.array(clu_new_all_hum["random"]),
    label="var-random"
)
plt.scatter(
    np.array(clu_new_all_met["avg"]) - np.array(clu_new_all_met["random"]),
    np.array(clu_new_all_hum["avg"]) - np.array(clu_new_all_hum["random"]),
    label="avg-random"
)
plt.legend()
plt.show()