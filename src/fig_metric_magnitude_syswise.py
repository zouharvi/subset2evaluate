import collections
import utils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import tqdm

data_old = utils.load_data()

# 'metric_SacreBLEU_bleu': 50.67309892897293, 'metric_SacreBLEU_chrf': 0.6685048422038385, 'metric_SacreBLEU_ter_neg': -0.5, 'metric_COMET': 0.810630738735199, 'metric_COMET_src': 0.00198473921045661, 'metric_Prism_ref': -1.248391628265381, 'metric_Prism_src': -3.505037784576416, 'metric_BERT_SCORE': 0.9561068415641785, 'metric_BLEURT_default': 0.5591212511062622, 'metric_BLEURT_large': 0.6688846945762634, 'metric_CharacTER_neg': -0.4268292682926829, 'metric_ESIM_': 0.8101189732551575}}
utils.matplotlib_default()
plt.figure(figsize=(3, 2))
points_x = []
points_y = []

# higher density at the beginning because there are higher changes in y
PROPS = np.concatenate([np.linspace(0, 0.04, 35), np.linspace(0.04, 1, 65)])

sys_to_lines = collections.defaultdict(list)
for line in data_old:
    for sys, sys_v in line.items():
        sys_to_lines[sys].append(sys_v)
# sort independently for each system
for sys, sys_v in sys_to_lines.items():
    sys_v.sort(key=lambda line: line["metrics"]["metric_COMET"])
    
for prop in tqdm.tqdm(PROPS):
    points_x.append(prop)

    if prop == 0.0:
        points_y.append(0.5)
    else:

        data_new = []
        for sys, sys_v in sys_to_lines.items():
            # pretend we have individual line each for one system
            for x in sys_v[:int(len(sys_v)*prop)]:
                data_new.append({sys: x})
        points_y.append(utils.eval_data_pairs(data_new, data_old))

plt.scatter(
    points_x, points_y,
    marker="o", s=10, label="From lowest"
)

points_x = []
points_y = []
for prop in tqdm.tqdm(PROPS):
    points_x.append(prop)

    if prop == 0.0:
        points_y.append(0.5)
    else:
        data_new = []
        for sys, sys_v in sys_to_lines.items():
            # pretend we have individual line each for one system
            for x in sys_v[-int(len(sys_v)*prop):]:
                data_new.append({sys: x})
        # taking lines with the highest metric score
        points_y.append(utils.eval_data_pairs(data_new, data_old))

plt.scatter(
    points_x, points_y,
    marker="o", s=10, label="From highest"
)


plt.ylabel("Sys. rank accuracy" + " "*5, labelpad=-5)
plt.xlabel("Proportion of original data", labelpad=-2)

plt.legend(frameon=False)

ax = plt.gca()
ax.spines[['top', 'right']].set_visible(False)
ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda y, _: '{:.0%}'.format(y))) 
ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda y, _: '{:.0%}'.format(y))) 

plt.tight_layout(pad=0.1)
plt.savefig("figures/metric_magnitude_syswise.png", dpi=200)
plt.savefig("figures/metric_magnitude_syswise.pdf")
plt.show()