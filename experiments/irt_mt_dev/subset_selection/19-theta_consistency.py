# %%

import os

import numpy as np
os.chdir("/home/vilda/irt-mt-dev")
import irt_mt_dev.utils as utils
import subset2evaluate.select_subset
import tqdm
import numpy as np
data_old = utils.load_data_wmt("wmt23", "en-cs", normalize=True)

def z_score(data):
    data = np.array(data)
    return (data - np.mean(data)) / np.std(data)

systems_gold_all = []
# run multiple times to average thetas
for _ in range(10):
    while True:
        try:        
            _data, params = subset2evaluate.select_subset.run_select_subset(
                data_old, method="pyirt_fic", metric="MetricX-23", irt_model="4pl_score", epochs=1000,
                return_model=True
            )
            break
        except:
            continue
    systems_k = list(params["systems"].keys())
    systems_v = z_score(list(params["systems"].values()))
    systems_gold_all.append(dict(zip(systems_k, systems_v)))

# %%

systems_k = list(systems_gold_all[0].keys())
systems_gold = {
    system: np.average([systems_gold[system] for systems_gold in systems_gold_all])
    for system in systems_k
}

# %%
acc_new_irt2irt = []
for prop in tqdm.tqdm(utils.PROPS):
    # get random data subset, importantly it's fixed for the same prop, even when we train multiple times
    data_random = subset2evaluate.select_subset.run_select_subset(
        data_old, method="random",
    )
    data_new = data_random[:int(len(data_old)*prop)]

    systems_pred_all = []
    for _ in range(10):
        while True:
            try:        
                # train IRT on the random subset
                _data, params = subset2evaluate.select_subset.run_select_subset(
                    data_new, method="pyirt_fic", metric="MetricX-23", irt_model="4pl_score", epochs=1000,
                    return_model=True
                )
                break
            except:
                pass

        systems_k = list(params["systems"].keys())
        systems_v = z_score(list(params["systems"].values()))
        systems_pred_all.append(dict(zip(systems_k, systems_v)))

    # average thetas from multiple rounds
    systems_k = list(systems_pred_all[0].keys())
    systems_pred = {
        system: np.average([systems_pred[system] for systems_pred in systems_pred_all])
        for system in systems_k
    }

    # see how close the thetas are to the total thetas
    acc_new_irt2irt.append(utils.eval_order_accuracy(systems_pred, systems_gold))

# %%
print(f"irt->irt: {np.average(np.max([np.array(acc_new_irt2irt), 1-np.array(acc_new_irt2irt)], axis=0)):.2%}")
# fix if some accuracies are 1-acc flipped
print(np.max([np.array(acc_new_irt2irt), 1-np.array(acc_new_irt2irt)], axis=0))

# %%

import subset2evaluate.evaluate
_, acc_new_random = subset2evaluate.evaluate.run_evaluate_topk(
    data_old,
    subset2evaluate.select_subset.run_select_subset(data_old, method="random"),
    metric="MetricX-23"
)
print(f"random: {np.average(acc_new_random):.2%}")

# RESULT:
# seems like the IRT is not modelling the competition well
# the subset consistency based on the IRT thetas is worse than consistency based on averages from random subset selection