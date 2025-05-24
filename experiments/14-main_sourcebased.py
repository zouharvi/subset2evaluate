# %%
import collections
import tqdm
import subset2evaluate.utils as utils
import subset2evaluate.evaluate
import subset2evaluate.select_subset
import numpy as np
import utils_fig

data_old_all = list(utils.load_data_wmt_test(normalize=True).items())

# %%

points_y_spa = collections.defaultdict(list)

# cache models because that's where we lose a lot of time
MODELS = {
    method: subset2evaluate.select_subset.basic(data_old_all[0][1], method=method, return_model=True)[1]
    for method in [
        "precomet_avg",
        "precomet_var",
        "precomet_diffdisc_direct",
        "precomet_diversity",
        "precomet_cons",
    ]
}
MODELS["random"] = None

for data_name, data_old in tqdm.tqdm(data_old_all):
    for repetitions, method in [
        (1, "precomet_avg"),
        (1, "precomet_var"),
        (1, "precomet_diffdisc_direct"),
        (1, "precomet_diversity"),
        (1, "precomet_cons"),
        (100, "random"),
    ]:
        for _ in range(repetitions):
            data_new = subset2evaluate.select_subset.basic(data_old, method=method, load_model=MODELS[method])
            spa_new = subset2evaluate.evaluate.eval_spa(data_new, data_old, metric="human")
            points_y_spa[method].append(spa_new)

points_y_spa_raw = points_y_spa.copy()

# %%


points_y_spa_random = np.average(np.array(points_y_spa_raw["random"]), axis=(0))
points_y_spa_random.sort(axis=0)
points_y_spa_random_075 = [
    utils.confidence_interval(np.array(points_y_spa_random)[:, i], confidence=0.75)
    for i in range(len(utils.PROPS))
]
points_y_spa_random_095 = [
    utils.confidence_interval(np.array(points_y_spa_random)[:, i], confidence=0.95)
    for i in range(len(utils.PROPS))
]

points_y_spa = {
    k: np.average(np.array(v), axis=(0, 1))
    for k, v in points_y_spa_raw.items()
}
# %%

utils_fig.plot_subset_selection(
    [
        (utils.PROPS, points_y_spa["random"], f"Random {np.average(points_y_spa['random']):.1%}"),
        (utils.PROPS, points_y_spa["precomet_avg"], f"MetricAvg$^\\mathrm{{src}}$ {np.average(points_y_spa['precomet_avg']):.1%}"),
        (utils.PROPS, points_y_spa["precomet_var"], f"MetricVar$^\\mathrm{{src}}$ {np.average(points_y_spa['precomet_var']):.1%}"),
        (utils.PROPS, points_y_spa['precomet_cons'], f"MetricCons$^\\mathrm{{src}}$ {np.average(points_y_spa['precomet_cons']):.1%}"),
        (utils.PROPS, points_y_spa['precomet_diversity'], f"Diversity$^\\mathrm{{src}}$ {np.average(points_y_spa['precomet_diversity']):.1%}"),
        (utils.PROPS, points_y_spa['precomet_diffdisc'], f"DiffDisc$^\\mathrm{{src}}$ {np.average(points_y_spa['precomet_diffdisc']):.1%}"),
    ],
    colors=["#000000"] + utils_fig.COLORS,
    filename="14-main_sourcebased",
    fn_extra=lambda ax: [
        ax.fill_between(
            range(len(utils.PROPS)),
            [x[0] for x in points_y_spa_random_095],
            [x[1] for x in points_y_spa_random_095],
            alpha=0.2,
            color="#000000",
            linewidth=0,
        ),

        ax.fill_between(
            range(len(utils.PROPS)),
            [x[0] for x in points_y_spa_random_075],
            [x[1] for x in points_y_spa_random_075],
            alpha=0.4,
            color="#000000",
            linewidth=0,
        ),
    ]
)