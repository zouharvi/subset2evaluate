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

data_old_all = list(utils.load_data_wmt_test(normalize=True).items())
data_old_name, data_old = data_old_all[args.dataset_i]

# %%
# precompute parity randnorm
spa_precomputed = subset2evaluate.evaluate.precompute_spa_randnorm(data_old)

# %%
os.makedirs("computed/16-metric_quality/", exist_ok=True)
with open(f"computed/16-metric_quality/precomputed_d{args.dataset_i}.pkl", "wb") as f:
    pickle.dump(spa_precomputed, f)


"""
for i in $(seq 0 8); do echo $i; python3 experiments/16b-metric_quality_precompute_spa.py $i; done
scp computed/16-metric_quality/precomputed_*.pkl euler:/cluster/work/sachan/vilem/subset2evaluate/computed/16-metric_quality/
"""