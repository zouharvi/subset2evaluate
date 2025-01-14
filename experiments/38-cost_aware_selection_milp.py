# %%

import tqdm
import subset2evaluate.utils as utils
import numpy as np
import subset2evaluate.evaluate
import subset2evaluate.select_subset
import scipy.optimize
import subset2evaluate.utils

data_old_all = list(utils.load_data_wmt_all(normalize=True).values())[:9]

for method_kwargs in tqdm.tqdm([
    dict(method="metric_avg", metric="MetricX-23"),
    # dict(method="metric_var", metric="MetricX-23"),
    dict(method="diversity_bleu"),
    dict(method="pyirt_diffdisc", metric="MetricX-23", retry_on_error=True),
]):
    clu_all = []
    acc_all = []
    for data_old in data_old_all:
        data_new_raw = subset2evaluate.select_subset.run_select_subset(data_old, **method_kwargs)
        # sort back
        data_new_raw.sort(key=lambda x: x["i"])
        assert all(["subset2evaluate_utility" in x for x in data_new_raw])

        # make sure utility is always positive
        data_new_utility = np.array([x["subset2evaluate_utility"] for x in data_new_raw])
        data_new_utility -= min(data_new_utility)
        data_new_utility += 1

        for prop in subset2evaluate.utils.PROPS:
            k = int(prop * len(data_old))
            # formulation
            # min c^T x
            # such that
            # A_ub x <= b_ub (upper bound)
            opt = scipy.optimize.milp(
                # minimize negative utility
                c=-data_new_utility,
                # 0 <= x <= 1
                bounds=scipy.optimize.Bounds(0, 1),
                constraints=scipy.optimize.LinearConstraint(
                    A=[line["time"] for line in data_new_raw],
                    # TODO: remove?
                    lb=0,
                    ub=k,
                ),
                # has to be integer
                integrality=np.full_like(data_new_raw, 1),
            )


            top_k = list(np.argsort(opt.x))
            data_new = []
            while True:
                new_line = data_new_raw[top_k.pop()]
                if sum([line["time"] for line in data_new+[new_line]]) >= k:
                    break
                data_new.append(new_line)
            # data_new = [line for x, line in zip(opt.x, data_new_raw) if x == 1.0]
            clu_new = subset2evaluate.evaluate.eval_subset_clusters(data_new)
            acc_new = subset2evaluate.evaluate.eval_subset_accuracy(data_new, data_old)

            clu_all.append(clu_new)
            acc_all.append(acc_new)

    print(f"{method_kwargs['method']:>15}", f"{np.average(clu_all):.2f} {np.average(acc_all):.1%}")