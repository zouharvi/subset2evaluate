
import irt_mt_dev.utils as utils
import numpy as np
import os
import subset2evaluate.evaluate
import subset2evaluate.select_subset
import tqdm
import sys
import json
import traceback

os.chdir("/home/vilda/irt-mt-dev")

def benchmark_method(repetitions=10, kwargs_dict={}):
    data_old_all = list(utils.load_data_wmt_all(normalize=True).items())[:9]
    points_y_acc = []
    points_y_clu = []

    # run multiple times to smooth variance
    for data_old_name, data_old in data_old_all:
        for _ in range(repetitions):
            print(f"Running {data_old_name}/{_+1}")
            clu_new, acc_new = subset2evaluate.evaluate.run_evaluate_topk(
                data_old,
                subset2evaluate.select_subset.run_select_subset(data_old, **kwargs_dict),
                metric="human",
                retry_on_error=True,
            )
            points_y_acc.append(acc_new)
            points_y_clu.append(clu_new)

    print(f"ACC: {np.average(points_y_acc):.2%} | CLU: {np.average(points_y_clu):.2f}")
    

kwargs = json.loads(sys.argv[1].replace("'", '"'))
benchmark_method(repetitions=2, kwargs_dict={**kwargs, "method": "pyirt_fic", "metric": "MetricX-23"})