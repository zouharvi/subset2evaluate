# %%

import matplotlib.pyplot as plt
import tqdm
import subset2evaluate.utils as utils
import numpy as np
import random
import subset2evaluate.evaluate
import subset2evaluate.select_subset
import scipy.stats

import subset2evaluate.utils

data_old = list(utils.load_data_wmt_all(normalize=True).values())[0]
sys_ord_true = subset2evaluate.evaluate.get_model_absolute(
    data_old, metric="human")

systems = list(data_old[0]["scores"].keys())
system_scores = {
    sys: {
        "scores": [],
        "scores_todo": [line["scores"][sys]["human"] for line in data_old],
        "ci": None,
    }
    for sys in systems
}

# randomly choose 10 items and evaluate all systems there
random_items = random.sample(range(len(data_old)), k=10)
for item in sorted(random_items, reverse=True):
    for sys in systems:
        # it's fine to pop because it's reverse
        score = system_scores[sys]["scores_todo"].pop(item)
        system_scores[sys]["scores"].append(score)

# compute confidence intervals
for sys in systems:
    scores = system_scores[sys]["scores"]
    system_scores[sys]["ci"] = scipy.stats.t.interval(
        0.9, len(scores), loc=np.mean(scores), scale=scipy.stats.sem(scores))


points_y = []
# do sampling
for budget in tqdm.tqdm(range(10*len(systems), len(data_old) * len(systems))):
    sys_available = [
        sys
        for sys in systems
        if system_scores[sys]["scores_todo"]
    ]

    # break if no system has any more items to evaluate
    if not sys_available:
        break
    # select system with largest CI that has some more items to evaluate
    # sys = max(
    #     sys_available,
    #     key=lambda x: system_scores[x]["ci"][1] - system_scores[x]["ci"][0]
    # )
    # select system that overlaps with most other systems
    sys = max(
        sys_available,
        key=lambda sys1: sum(
            [
                1
                for sys2 in systems
                if sys2 != sys1
                and system_scores[sys1]["ci"][0] < system_scores[sys2]["ci"][1]
                and system_scores[sys1]["ci"][1] > system_scores[sys2]["ci"][0]
            ]
        ),
    )
    # randomly choose and pop item
    item = random.choice(range(len(system_scores[sys]["scores_todo"])))
    score = system_scores[sys]["scores_todo"].pop(item)
    system_scores[sys]["scores"].append(score)
    # update CI
    scores = system_scores[sys]["scores"]
    system_scores[sys]["ci"] = scipy.stats.t.interval(0.9, len(scores), loc=np.mean(scores), scale=scipy.stats.sem(scores))
    # compute average for each system
    sys_ord_tmp = {
        sys: np.mean(system_scores[sys]["scores"])
        for sys in systems
    }
    points_y.append((budget, scipy.stats.spearmanr(
        list(sys_ord_true.values()), list(sys_ord_tmp.values())).correlation))

# %%


points_y_true = []
props_k = [int(p * len(data_old)) for p in subset2evaluate.utils.PROPS]
for budget, corr in points_y:
    budget_true = budget / len(systems)
    # check is integer
    if budget_true.is_integer() and int(budget_true) in props_k:
        points_y_true.append(corr)


plt.plot(
    range(len(subset2evaluate.utils.PROPS)),
    points_y_true,
    label=f"{np.average(points_y_true):.1%}",
)
plt.legend()
plt.show()


# random item sampling: 85.5%
# select system with highest CI overlap. 89.4%

# %%
# compare to random baseline
cor_all = []
clu_all = []
for _ in range(100):
    clu_new, cor_new = subset2evaluate.evaluate.eval_clucor(
        subset2evaluate.select_subset.basic(data_old, method="random"),
        data_old
    )
    clu_all.append(clu_new)
    cor_all.append(cor_new)
print("Random baseline:", f"{np.average(clu_all):.2f}", f"{np.average(cor_all):.1%}")