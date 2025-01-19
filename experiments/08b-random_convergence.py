import subset2evaluate.utils as utils
import utils_fig
import random
import numpy as np
import tqdm
import scipy.stats as st
import matplotlib.pyplot as plt
import subset2evaluate.evaluate

data_old = utils.load_data_wmt()


def confidence_interval(data):
    return st.t.interval(
        confidence=0.95,
        df=len(data) - 1,
        loc=np.mean(data),
        scale=np.std(data)
    )


points_x = []
points_y_struct = []

_random = random.Random(0)

for prop in tqdm.tqdm(utils.PROPS):
    k = int(len(data_old) * prop)
    points_x.append(k)

    points_y_local = []
    # repeat each sampling 10 times to smooth it out
    for _ in range(10):
        data_new = _random.sample(data_old, k=k)
        points_y_local.append(subset2evaluate.evaluate.get_model_absolute(data_new))

    points_y_struct.append(points_y_local)

models = list(points_y_struct[0][0].keys())
models_highlighted = random.Random(3).sample(models, k=5)

# take the first run
points_y_single = [points_y_local[0] for points_y_local in points_y_struct]
points_y_interval = [
    {
        model: confidence_interval([x[model] for x in points_y_local])
        for model in models
    }
    for points_y_local in points_y_struct
]

utils_fig.matplotlib_default()
plt.figure(figsize=(6, 2.5))

for model in models:
    plt.plot(
        points_x,
        [x[model] for x in points_y_single],
        marker=".",
        markersize=10,
        color="black",
        clip_on=False,
        alpha=1 if model in models_highlighted else 0.25,
        linewidth=3,
    )
    plt.fill_between(
        points_x,
        [x[model][0] for x in points_y_interval],
        [x[model][1] for x in points_y_interval],
        color=utils_fig.COLORS[1],
        alpha=0.2,
        linewidth=0,
    )
for model in models_highlighted:
    plt.text(
        x=points_x[-1] + 10,
        y=points_y_single[-1][model],
        s=model.replace("DocTransformer", "Doc"),
        fontsize=8,
        ha="left", va="center",
    )

plt.ylabel("Human average", labelpad=-1)
plt.xlabel("Sentences subset size", labelpad=-1)

ax = plt.gca()
ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout(pad=0.2)
plt.savefig("figures/08b-random_convergence.svg")
plt.show()
