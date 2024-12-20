# %%
import collections

import tqdm
import irt_mt_dev.utils as utils
import subset2evaluate.evaluate
import subset2evaluate.select_subset
import numpy as np
import os
import pickle
import argparse

args = argparse.ArgumentParser()
args.add_argument("i", type=int)
args = args.parse_args()

# 33 items
data_old = list(utils.load_data_wmt_all(normalize=True).values())[args.i]
data_train = collections.defaultdict(list)
data_params = []

# train IRT on all data, distinguish where it came from
# run multiple times and average item parameters
for _ in tqdm.tqdm(range(5)):
    _data, params = subset2evaluate.select_subset.run_select_subset(data_old, return_model=True, method="pyirt_diffdisc", model="4pl_score", metric="human", epochs=1000, retry_on_error=True, enforce_positive_disc=True)
    data_params.append([
        {**line, "irt": line_irt}
        for line, line_irt in zip(data_old, params["items"])
    ])

def average_irt_params(data_train):
    # the old data (e.g. line src) is the same everywhere
    data_new = [l for l in data_train[0]]
    irt_params = [collections.defaultdict(list) for _ in data_train]
    for i in range(len(data_train)):
        for data in data_train:
            for k, v in data[i]["irt"].items():
                irt_params[i][k].append(v)

    for i in range(len(data_train)):
        for k, v in irt_params[i].items():
            # TODO: try median?
            data_new[i]["irt"][k] = np.average(v)

    return data_new

data_params = average_irt_params(data_params)


# dump to computed/irt_params
os.makedirs("computed/irt_params", exist_ok=True)
pickle.dump(data_params, open(f"computed/irt_params/{args.i}.pkl", "wb"))



# for i in $(seq 0 32); do
#     sbatch_gpu_short "irt_params_$i" "python3 experiments/irt_mt_dev/subset_selection/26-get_irt_params.py $i"
# done;
