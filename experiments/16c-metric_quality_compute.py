# %%
import collections
import subset2evaluate.select_subset
import subset2evaluate.evaluate
import subset2evaluate.utils
import numpy as np
import scipy.stats
import subset2evaluate.utils as utils
import pickle
import argparse
import tqdm
import os

args = argparse.ArgumentParser()
args.add_argument("dataset_i", type=int, default=0)
args.add_argument("metric_i", type=int, default=0)
args = args.parse_args()

print("Start loading data", flush=True)
data_old_all = list(utils.load_data_wmt_test(normalize=True).items())
data_old_name, data_old = data_old_all[args.dataset_i]
print("Finish loading data", flush=True)

# WMT23 has 35-44 metrics
# print({
#     k: len(list(v[0]["scores"].values())[0].keys())
#     for k, v in data_old_all
# })
# exit()

# %%
# precompute parity randnorm
with open(f"computed/16-metric_quality/precomputed_d{args.dataset_i}.pkl", "rb") as f:
    clucor_precomputed = pickle.load(f)
print("Loaded precomputed clucor", flush=True)

# %%
models = list(data_old[0]["scores"].keys())
data_y_human = [
    line["scores"][model]["human"]
    for line in data_old
    for model in models
]
metrics = list(list(data_old[0]["scores"].values())[0].keys())
metrics = [metric for metric in metrics if metric != "human"]
if args.metric_i >= len(metrics):
    print(f"No metric_i {args.metric_i} in {data_old_name}")
    exit()
metric = metrics[args.metric_i]

data_y_metric = [
    line["scores"][model][metric]
    for line in data_old
    for model in models
]
metric_corr = scipy.stats.pearsonr(data_y_human, data_y_metric)[0]
result_item = {
    "wmt": data_old_name,
    "metric": metric,
    "correlation": metric_corr,
    "spa": collections.defaultdict(list),
}

for repetitions, method_kwargs in tqdm.tqdm([
    (1, dict(method="metric_avg", metric=metric)),
    (1, dict(method="metric_var", metric=metric)),
    (1, dict(method="metric_cons", metric=metric)),
    (1, dict(method="diversity", metric="LM")),
    (5, dict(method="pyirt_diffdisc", metric=metric, model="4pl_score", retry_on_error=True)),
]):
    clus_local = []
    cors_local = []
    for _ in range(repetitions):
        print("Computing", method_kwargs["method"], "on", metric, flush=True)
        data_new_avg = subset2evaluate.select_subset.basic(data_old, **method_kwargs)
        print("Evaluating", method_kwargs["method"], "on", metric, flush=True)
        spa_new = subset2evaluate.evaluate.eval_spa_par_randnorm(
            data_new_avg, data_old,
            spa_precomputed=clucor_precomputed,
        )
        clus_local.append(spa_new)
    result_item["spa"][method_kwargs["method"]] = np.average(clus_local)

# %%
os.makedirs("computed/16-metric_quality/", exist_ok=True)
with open(f"computed/16-metric_quality/d{args.dataset_i}_m{args.metric_i}.pkl", "wb") as f:
    pickle.dump(result_item, f)

"""
sbatch_gpu_short "metric_quality_d0_m0" "python3 experiments/16c-metric_quality_compute.py 0 0"

for di in $(seq 0 8); do
    # for mi in $(seq 0 25); do
    for mi in $(seq 26 44); do
        sbatch_gpu_short "metric_quality_d${di}_m${mi}" "python3 experiments/16c-metric_quality_compute.py ${di} ${mi}"
    done;
done;
"""