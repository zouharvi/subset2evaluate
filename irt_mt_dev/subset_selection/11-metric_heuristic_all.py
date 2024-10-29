import irt_mt_dev.utils as utils
import irt_mt_dev.utils.fig
import numpy as np
import tqdm
import scipy.stats
import itertools

data_old = utils.load_data()

points_x = []
points_y_metricx_avg_acc = []
points_y_metricx_avg_clu = []
points_y_chrf_avg_acc = []
points_y_chrf_avg_clu = []
points_y_metricx_var_acc = []
points_y_metricx_var_clu = []

# mre-score-labse-regular', 'MetricX-23', 'chrF', 'COMET', 'f200spBLEU', 'tokengram_F', 'YiSi-1', 'embed_llama', 'XCOMET-XXL', 'BLEU', 'prismRef', 'eBLEU', 'XCOMET-XL', 'MetricX-23-c', 'XCOMET-Ensemble', 'BERTscore', 'XLsim', 'BLEURT-20', 'MetricX-23-b'
def heuristic_metricx_avg(line):
    return np.average(
        [sys_v["MetricX-23"] for sys_v in line["scores"].values()]
    )

def heuristic_metricx_var(line):
    return np.var([sys_v["MetricX-23"] for sys_v in line["scores"].values()])

def heuristic_chrf_avg(line):
    return np.average(
        [sys_v["chrF"] for sys_v in line["scores"].values()]
    )

def heuristic_chrf_var(line):
    return np.var([sys_v["chrF"] for sys_v in line["scores"].values()])

# sort by the heuristic
data_metricx_avg = sorted(data_old, key=lambda x: heuristic_metricx_avg(x))
data_metricx_var = sorted(data_old, key=lambda x: -heuristic_metricx_var(x))
data_chrf_avg = sorted(data_old, key=lambda x: heuristic_chrf_avg(x))

for prop in tqdm.tqdm(utils.PROPS):
    points_x.append(prop)

    points_y_metricx_avg_acc.append(
        utils.eval_order_accuracy(data_metricx_avg[: int(len(data_old) * prop)], data_old)
    )
    points_y_metricx_avg_clu.append(
        utils.eval_system_clusters(data_metricx_avg[: int(len(data_old) * prop)])
    )
    points_y_chrf_avg_acc.append(
        utils.eval_order_accuracy(data_chrf_avg[: int(len(data_old) * prop)], data_old)
    )
    points_y_chrf_avg_clu.append(
        utils.eval_system_clusters(data_chrf_avg[: int(len(data_old) * prop)])
    )
    points_y_metricx_var_acc.append(
        utils.eval_order_accuracy(data_metricx_var[: int(len(data_old) * prop)], data_old)
    )
    points_y_metricx_var_clu.append(
        utils.eval_system_clusters(data_metricx_var[: int(len(data_old) * prop)])
    )

irt_mt_dev.utils.fig.plot_subset_selection(
    points=[
        (points_x, points_y_chrf_avg_acc, f"ChrF average {np.average(points_y_chrf_avg_acc):.2%}"),
        (points_x, points_y_metricx_avg_acc, f"MetricX average {np.average(points_y_metricx_avg_acc):.2%}"),
        (points_x, points_y_metricx_var_acc, f"MetricX variance {np.average(points_y_metricx_var_acc):.2%}"),
    ],
    filename="11-metric_heuristic_all",
)

irt_mt_dev.utils.fig.plot_subset_selection(
    points=[
        (points_x, points_y_chrf_avg_clu, f"ChrF average {np.average(points_y_chrf_avg_clu):.2f}"),
        (points_x, points_y_metricx_avg_clu, f"MetricX average {np.average(points_y_metricx_avg_clu):.2f}"),
        (points_x, points_y_metricx_var_clu, f"MetricX variance {np.average(points_y_metricx_var_clu):.2f}"),
    ],
    filename="11-metric_heuristic_all",
)