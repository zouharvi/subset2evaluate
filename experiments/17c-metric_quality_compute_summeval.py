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
# SummEval has 119 metrics
args.add_argument("metric_i", type=int, default=0)
args = args.parse_args()

metric_target = ["human_relevance", "human_coherence", "human_consistency", "human_fluency", "human_sum"][args.dataset_i]
PROPS = np.geomspace(0.25, 0.75, 10)

# %%
print("Start loading data", flush=True)
data_old = utils.load_data_summeval(normalize=True, load_extra=True)
print("Finish loading data", flush=True)

# %%
# precompute parity randnorm
with open(f"computed/17-metric_quality/precomputed_d{args.dataset_i}.pkl", "rb") as f:
    clucor_precomputed = pickle.load(f)
print("Loaded precomputed clucor", flush=True)

# %%
models = list(data_old[0]["scores"].keys())
data_y_human = [
    line["scores"][model][metric_target]
    for line in data_old
    for model in models
]
metrics = list(list(data_old[0]["scores"].values())[0].keys())
metrics = [metric for metric in metrics if not metric.startswith("human_")]
print(len(metrics))

if args.metric_i >= len(metrics):
    print(f"No metric_i {args.metric_i}")
    exit()
metric = metrics[args.metric_i]

data_y_metric = [
    line["scores"][model][metric]
    for line in data_old
    for model in models
]
metric_corr = scipy.stats.pearsonr(data_y_human, data_y_metric)[0]
result_item = {
    "metric_target": metric_target,
    "metric": metric,
    "correlation": metric_corr,
    "cor": collections.defaultdict(list),
    "clu": collections.defaultdict(list),
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
        clu_new, cor_new = subset2evaluate.evaluate.eval_clucor_par_randnorm(
            data_new_avg, data_old,
            clucor_precomputed=clucor_precomputed,
            metric=metric_target,
            props=PROPS,
        )
        clus_local.append(clu_new)
        cors_local.append(cor_new)
    result_item["clu"][method_kwargs["method"]] = np.average(clus_local)
    result_item["cor"][method_kwargs["method"]] = np.average(cors_local)

# %%
os.makedirs("computed/17-metric_quality/", exist_ok=True)
with open(f"computed/17-metric_quality/d{args.dataset_i}_m{args.metric_i}.pkl", "wb") as f:
    pickle.dump(result_item, f)

"""
sbatch_gpu_short_small "summeval_metric_quality_d0_m0" "python3 experiments/17c-metric_quality_compute_summeval.py 0 0"

for di in $(seq 0 4); do
    for mi in $(seq 0 118); do
        echo "summeval_metric_quality_d${di}_m${mi}"
        sbatch_gpu_short_small "summeval_metric_quality_d${di}_m${mi}" "python3 experiments/17c-metric_quality_compute_summeval.py ${di} ${mi}"
    done;
done;
"""