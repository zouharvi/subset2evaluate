# %%

import collections
import subset2evaluate.utils as utils
import utils_fig
import numpy as np
import tqdm

data_old_all = list(utils.load_data_wmt_all().values())[:9]

points_y_acc_all = collections.defaultdict(list)
points_y_clu_all = collections.defaultdict(list)

def heuristic_metricx_avg(line):
    return np.average(
        [sys_v["MetricX-23-c"] for sys_v in line["scores"].values()]
    )

def heuristic_metricx_var(line):
    return np.var([sys_v["MetricX-23-c"] for sys_v in line["scores"].values()])

def heuristic_chrf_avg(line):
    return np.average(
        [sys_v["chrF"] for sys_v in line["scores"].values()]
    )

def heuristic_chrf_var(line):
    return np.var([sys_v["chrF"] for sys_v in line["scores"].values()])

for data_old in tqdm.tqdm(data_old_all):
    # sort by the heuristic
    data_metricx_avg = sorted(data_old, key=lambda x: heuristic_metricx_avg(x))
    data_metricx_var = sorted(data_old, key=lambda x: -heuristic_metricx_var(x))
    data_chrf_avg = sorted(data_old, key=lambda x: heuristic_chrf_avg(x))

    points_y_acc = collections.defaultdict(list)
    points_y_clu = collections.defaultdict(list)

    for prop in utils.PROPS:

        points_y_acc["metricx_avg"].append(
            utils.eval_subset_accuracy(data_metricx_avg[: int(len(data_old) * prop)], data_old)
        )
        points_y_clu["metricx_avg"].append(
            utils.eval_system_clusters(data_metricx_avg[: int(len(data_old) * prop)])
        )
        points_y_acc["chrf_avg"].append(
            utils.eval_subset_accuracy(data_chrf_avg[: int(len(data_old) * prop)], data_old)
        )
        points_y_clu["chrf_avg"].append(
            utils.eval_system_clusters(data_chrf_avg[: int(len(data_old) * prop)])
        )
        points_y_acc["metricx_var"].append(
            utils.eval_subset_accuracy(data_metricx_var[: int(len(data_old) * prop)], data_old)
        )
        points_y_clu["metricx_var"].append(
            utils.eval_system_clusters(data_metricx_var[: int(len(data_old) * prop)])
        )
    
    # add lists to the global list
    for k, v in points_y_acc.items():
        points_y_acc_all[k].append(v)
    for k, v in points_y_clu.items():
        points_y_clu_all[k].append(v)

points_y_acc_all = {
    k: np.average(np.array(v), axis=0)
    for k,v in points_y_acc_all.items()
}
points_y_clu_all = {
    k: np.average(np.array(v), axis=0)
    for k,v in points_y_clu_all.items()
}
# %%
utils_fig.plot_subset_selection(
    points=[
        (utils.PROPS, points_y_acc_all["chrf_avg"], f"ChrF average {np.average(points_y_acc_all['chrf_avg']):.2%}"),
        (utils.PROPS, points_y_acc_all["metricx_avg"], f"MetricX average {np.average(points_y_acc_all['metricx_avg']):.2%}"),
        (utils.PROPS, points_y_acc_all["metricx_var"], f"MetricX variance {np.average(points_y_acc_all['metricx_var']):.2%}"),
    ],
    filename="13-main_outputbased_metrics_moment",
)


utils_fig.plot_subset_selection(
    points=[
        (utils.PROPS, points_y_clu_all["chrf_avg"], f"ChrF average {np.average(points_y_clu_all['chrf_avg']):.2f}"),
        (utils.PROPS, points_y_clu_all["metricx_avg"], f"MetricX average {np.average(points_y_clu_all['metricx_avg']):.2f}"),
        (utils.PROPS, points_y_clu_all["metricx_var"], f"MetricX variance {np.average(points_y_clu_all['metricx_var']):.2f}"),
    ],
    filename="13-main_outputbased_metrics_moment",
)