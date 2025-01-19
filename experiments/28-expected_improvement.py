# %%
import collections
import subset2evaluate.utils as utils
import subset2evaluate.evaluate
import subset2evaluate.select_subset
import numpy as np
import tqdm
import matplotlib.pyplot as plt

data_old_all = list(utils.load_data_wmt_all(normalize=True).values())[:9]

# %%
# aggregate scores

cor_new_all_hum = collections.defaultdict(list)
clu_new_all_hum = collections.defaultdict(list)
cor_new_all_met = collections.defaultdict(list)
clu_new_all_met = collections.defaultdict(list)

for data_old in tqdm.tqdm(data_old_all):
    for method_kwargs in [
        dict(method="random"),
        dict(method="metric_avg"),
        dict(method="metric_var"),
        dict(method="diversity_bleu"),
        # dict(method="pyirt_diffdisc", model="4pl_score", epochs=1000),
    ]:
        # run multiple times to average out the effect
        cor_new_all_hum_local = []
        clu_new_all_hum_local = []
        cor_new_all_met_local = []
        clu_new_all_met_local = []
        for _ in range(5 if method_kwargs["method"] == "pyirt_diffdisc" else 10 if method_kwargs["method"] == "random" else 1):
            data_new = subset2evaluate.select_subset.basic(
                data_old,
                **method_kwargs,
                metric="MetricX-23-c",
                retry_on_error=True,
            )
            clu_new, cor_new = subset2evaluate.evaluate.eval_clucor(data_new, data_old, metric="human")
            cor_new_all_hum_local.append(np.average(cor_new))
            clu_new_all_hum_local.append(np.average(clu_new))

            clu_new, cor_new = subset2evaluate.evaluate.eval_clucor(data_new, data_old, metric="MetricX-23-c")
            cor_new_all_met_local.append(np.average(cor_new))
            clu_new_all_met_local.append(np.average(clu_new))

        cor_new_all_hum[method_kwargs["method"]].append(np.average(cor_new_all_hum_local))
        clu_new_all_hum[method_kwargs["method"]].append(np.average(clu_new_all_hum_local))
        cor_new_all_met[method_kwargs["method"]].append(np.average(cor_new_all_met_local))
        clu_new_all_met[method_kwargs["method"]].append(np.average(clu_new_all_met_local))


# %%

plt.scatter(
    clu_new_all_met["random"],
    clu_new_all_hum["random"],
    label="random"
)
plt.scatter(
    clu_new_all_met["metric_var"],
    clu_new_all_hum["metric_var"],
    label="metric_var"
)
plt.legend()
plt.show()

# %%

plt.scatter(
    np.array(clu_new_all_met["metric_var"]) - np.array(clu_new_all_met["random"]),
    np.array(clu_new_all_hum["metric_var"]) - np.array(clu_new_all_hum["random"]),
    label="var-random"
)
plt.scatter(
    np.array(clu_new_all_met["metric_avg"]) - np.array(clu_new_all_met["random"]),
    np.array(clu_new_all_hum["metric_avg"]) - np.array(clu_new_all_hum["random"]),
    label="avg-random"
)
plt.legend()
plt.show()
