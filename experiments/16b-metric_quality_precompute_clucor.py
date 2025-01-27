# %%
import subset2evaluate.select_subset
import subset2evaluate.evaluate
import subset2evaluate.utils
import subset2evaluate.utils as utils
import pickle
import argparse
import os

args = argparse.ArgumentParser()
args.add_argument("dataset_i", type=int, default=0)
args = args.parse_args()

data_old_all = list(utils.load_data_wmt_all(normalize=True).items())
data_old_name, data_old = data_old_all[args.dataset_i]

# %%
# precompute parity randnorm
clucor_precomputed = subset2evaluate.evaluate.precompute_randnorm(data_old)

# %%
os.makedirs("computed/16-metric_quality/", exist_ok=True)
with open(f"computed/16-metric_quality/precomputed_d{args.dataset_i}.pkl", "wb") as f:
    pickle.dump(clucor_precomputed, f)


"""
for i in $(seq 0 8); do echo $i; python3 experiments/16b-metric_quality_precompute_clucor.py $i; done
scp computed/16-metric_quality/precomputed_*.pkl euler:/cluster/work/sachan/vilem/subset2evaluate/computed/16-metric_quality/
"""