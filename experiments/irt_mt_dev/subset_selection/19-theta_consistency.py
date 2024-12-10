# %%

import os

import numpy as np

import subset2evaluate.utils
os.chdir("/home/vilda/irt-mt-dev")
import irt_mt_dev.utils as utils
import subset2evaluate.select_subset
import tqdm
import numpy as np
data_old = utils.load_data_wmt("wmt23", "en-de", normalize=True)


systems_gold_all = []
# run multiple times to average thetas
for _ in tqdm.tqdm(range(20)):
    while True:
        try:        
            _data, params = subset2evaluate.select_subset.run_select_subset(
                data_old, method="pyirt_fic", metric="human", irt_model="4pl_score", epochs=1000,
                return_model=True
            )
            break
        except Exception as e:
            print(e)
            continue
    systems_gold_all.append(params["systems"])

# %%

def z_score(data):
    data = np.array(data)
    return (data - np.mean(data)) / np.std(data)

def align_merge_multiple_orderings(systems_all):
    # align all in the direction of the first
    systems_k0 = list(systems_all[0].keys())
    systems_v0 = [systems_all[0][system] for system in systems_k0]

    systems_all_local = []
    for systems_v in systems_all:
        systems_v = np.array([systems_v[system] for system in systems_k0])
        # align in the same direction if corrcoef < 0
        if np.corrcoef(systems_v0, systems_v)[0,1] < 0:
            systems_v = -systems_v
        systems_all_local.append(dict(zip(systems_k0, z_score(systems_v) )))


    return {
        system: np.average([systems_v[system] for systems_v in systems_all_local])
        for system in systems_k0
    }

systems_gold_ord = align_merge_multiple_orderings(systems_gold_all)
print(systems_gold_ord)

# %%
systems_pred_all_multi = []
for prop in tqdm.tqdm(utils.PROPS):
    # get random data subset, importantly it's fixed for the same prop, even when we train multiple times
    data_random = subset2evaluate.select_subset.run_select_subset(
        data_old, method="random",
    )
    data_new = data_random[:int(len(data_old)*prop)]

    systems_pred_all= []
    for _ in range(20):
        while True:
            try:
                # train IRT on the random subset
                _data, params = subset2evaluate.select_subset.run_select_subset(
                    data_new, method="pyirt_fic", metric="human", irt_model="4pl_score", epochs=1000,
                    return_model=True
                )
                break
            except Exception as e:
                print(e)
                continue

        systems_pred_all.append(params["systems"])
    systems_pred_all_multi.append(systems_pred_all)

# %%
import scipy.stats

# average thetas from multiple rounds
acc_new_rand2irt = []
corr_new_rand2irt = []
for systems_pred_all in systems_pred_all_multi:
    systems_pred_ord = align_merge_multiple_orderings(systems_pred_all)

    corr = np.abs(scipy.stats.pearsonr(list(systems_pred_ord.values()), list(systems_gold_ord.values())).correlation)
    corr_new_rand2irt.append(corr)
    # see how close the thetas are to the total thetas
    acc_new_rand2irt.append(utils.eval_order_accuracy(systems_pred_ord, systems_gold_ord))

# fix if some accuracies are 1-acc flipped
print(f"Random->IRT ACC: {np.average(np.max([np.array(acc_new_rand2irt), 1-np.array(acc_new_rand2irt)], axis=0)):.2%}")
print(f"Random->IRT CORR: {np.average(corr_new_rand2irt):.2%}")

# %%

import subset2evaluate.evaluate
acc_new_random_all = []
corr_new_random_all = []
systems_gold_ord_rand = utils.get_sys_absolute(metric="human", data_new=data_old)

for _ in range(100):
    data_new = subset2evaluate.select_subset.run_select_subset(data_old, method="random")
    _, acc_new_random = subset2evaluate.evaluate.run_evaluate_topk(
        data_old, data_new, metric="human"
    )

    corr_new_rand_local = []
    for prop in utils.PROPS:
        systems_pred_ord = utils.get_sys_absolute(metric="human", data_new=data_new[:int(len(data_old)*prop)])
        corr_new_rand_local.append(np.abs(scipy.stats.pearsonr(list(systems_pred_ord.values()), list(systems_gold_ord.values())).correlation))
    corr_new_random_all.append(corr_new_rand_local)     

    corr = np.abs(scipy.stats.pearsonr(list(systems_pred_ord.values()), list(systems_gold_ord.values())).correlation)

    # fix if some accuracies are 1-acc flipped
    acc_new_random_all.append(np.average(np.max([np.array(acc_new_random), 1-np.array(acc_new_random)], axis=0)))

print(f"random ACC: {np.average(acc_new_random_all):.2%}")
print(f"random CORR: {np.average(corr_new_random_all):.2%}" )

# RESULT:
# seems like the IRT is not modelling the competition well
# the subset consistency based on the IRT thetas is worse than consistency based on averages from random subset selection

# ACTUALLY
# on spearmanr and pearsonr, the IRT does slightly better!