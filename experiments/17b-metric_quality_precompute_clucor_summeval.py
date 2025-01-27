# %%
import subset2evaluate.select_subset
import subset2evaluate.evaluate
import subset2evaluate.utils
import subset2evaluate.utils as utils
import pickle
import argparse
import os
import numpy as np

args = argparse.ArgumentParser()
args.add_argument("dataset_i", type=int, default=0)
args = args.parse_args()

data_old = utils.load_data_summeval(normalize=True, load_extra=True)
PROPS = np.geomspace(0.25, 0.75, 10)
metric_target = ["human_relevance", "human_coherence", "human_consistency", "human_fluency", "human_sum"][args.dataset_i]

# %%
# precompute parity randnorm
clucor_precomputed = subset2evaluate.evaluate.precompute_randnorm(data_old, metric=metric_target, props=PROPS)

# %%
os.makedirs("computed/17-metric_quality/", exist_ok=True)
with open(f"computed/17-metric_quality/precomputed_d{args.dataset_i}.pkl", "wb") as f:
    pickle.dump(clucor_precomputed, f)


"""
for i in $(seq 0 4); do echo $i; python3 experiments/17b-metric_quality_precompute_clucor_summeval.py $i; done
scp computed/17-metric_quality/precomputed_*.pkl euler:/cluster/work/sachan/vilem/subset2evaluate/computed/17-metric_quality/
"""