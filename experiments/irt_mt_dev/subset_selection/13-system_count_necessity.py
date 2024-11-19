# %%
import subset2evaluate.select_subset
import subset2evaluate.evaluate
import subset2evaluate.utils
import copy
import random
import tqdm
import numpy as np
import multiprocessing
import irt_mt_dev.utils.fig as fig_utils
import matplotlib.pyplot as plt
import os
os.chdir("/home/vilda/irt-mt-dev")

random.seed(0)

data_old = subset2evaluate.utils.load_data("wmt23/en-cs")
data_old_i_to_line = {line["i"]: line for line in data_old}
systems = list(data_old[0]["scores"].keys())

# %%
accs_all = []
for subset_size in tqdm.tqdm(range(1, len(systems)+1)):
    def _run(_):
        systems_local = random.sample(systems, k=subset_size)
        data_old_local = copy.deepcopy(data_old)
        data_old_local = [
            {
                **line,
                "scores": {
                    sys: v
                    for sys, v in line["scores"].items()
                    if sys in systems_local
                }
            }
            for line in data_old_local
        ]

        data_new_avg = subset2evaluate.select_subset.run_select_subset(data_old_local, method="avg", metric="MetricX-23")
        data_new_var = subset2evaluate.select_subset.run_select_subset(data_old_local, method="var", metric="MetricX-23")
        data_new_irt = subset2evaluate.select_subset.run_select_subset(data_old_local, method="irt_ic", model="scalar", metric="MetricX-23")

        # we dropped some systems but we can recover them with the same ordering from data_old
        _, acc_new_avg = subset2evaluate.evaluate.run_evaluate_topk(data_old, [
            data_old_i_to_line[line["i"]]
            for line in data_new_avg
        ])
        _, acc_new_var = subset2evaluate.evaluate.run_evaluate_topk(data_old, [
            data_old_i_to_line[line["i"]]
            for line in data_new_var
        ])
        _, acc_new_irt = subset2evaluate.evaluate.run_evaluate_topk(data_old, [
            data_old_i_to_line[line["i"]]
            for line in data_new_irt
        ])
        return np.average(acc_new_avg), np.average(acc_new_var), np.average(acc_new_irt)
    
    acc_new = [_run(_) for _ in range(2)]

    # NOTE: can't use torch in multiprocessing
    # with multiprocessing.Pool(20) as pool:
    #     acc_new = pool.map(_run, range(20))
    accs_all.append(np.average(acc_new, axis=0))

# %%
fig_utils.matplotlib_default()
plt.plot([x[0] for x in accs_all], label="MetricX-23 avg")
plt.plot([x[1] for x in accs_all], label="MetricX-23 var")
plt.plot([x[2] for x in accs_all], label="IRT")
# plt.ylim(0.91, None)
plt.ylabel("Average accuracy")
plt.xlabel("Number of systems")
plt.legend()
plt.show()