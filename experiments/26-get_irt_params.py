# %%
import collections
import tqdm
import subset2evaluate.utils
import subset2evaluate.evaluate
import subset2evaluate.select_subset
import numpy as np
import os
import pickle
import argparse

args = argparse.ArgumentParser()
args.add_argument("split", choices=["train", "test"])
args.add_argument("i", type=int)
args = args.parse_args()

data_test = subset2evaluate.utils.load_data_wmt_test().keys()
data_all = subset2evaluate.utils.load_data_wmt_all(normalize=True, min_items=400).items()
data_all = [
    (data_name, data_local)
    for data_name, data_local in data_all
    if (data_name in data_test) == (args.split == "test")
]


# 49 items for train
# 9 items for test
data_old_name, data_old = data_all[args.i]
data_train = collections.defaultdict(list)
data_params = []

# train IRT on all data, distinguish where it came from
# run multiple times and average item parameters
for _ in tqdm.tqdm(range(5)):
    _data, params = subset2evaluate.select_subset.basic(data_old, return_model=True, method="pyirt_diffdisc", model="4pl_score", metric="human", epochs=1000, retry_on_error=True, enforce_positive_disc=True)
    data_params.append([
        {**line, "irt": line_irt}
        for line, line_irt in zip(data_old, params["items"])
    ])


def average_irt_params(data_train):
    # the old data (e.g. line src) is the same everywhere
    data_new = data_train[0]
    irt_params = [collections.defaultdict(list) for _ in data_train]
    for i in range(len(data_train)):
        for data in data_train:
            for k, v in data[i]["irt"].items():
                irt_params[i][k].append(v)

    for i in range(len(data_train)):
        for k, v in irt_params[i].items():
            data_new[i]["irt"][k] = np.average(v)

    return data_new


data_params = average_irt_params(data_params)

# dump to computed/irt_params
os.makedirs("computed/irt_params", exist_ok=True)
with open(f"computed/irt_params/{args.split}_{args.i}.pkl", "wb") as f:
    pickle.dump((data_old_name, data_params), f)

"""
for i in $(seq 0 48); do
    sbatch_gpu_short "irt_params_train_$i" "python3 experiments/26-get_irt_params.py train $i"
done;
for i in $(seq 0 8); do
    sbatch_gpu_short "irt_params_test_$i"  "python3 experiments/26-get_irt_params.py test $i"
done;
"""