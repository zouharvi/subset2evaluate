# %%

import subset2evaluate
import subset2evaluate.evaluate
import subset2evaluate.utils
import subset2evaluate.select_subset
import numpy as np
import collections
import tqdm

data_old = subset2evaluate.utils.load_data('summeval')
PROPS = np.geomspace(0.25, 0.75, 10)

results = collections.defaultdict(dict)

for metric_target in tqdm.tqdm(["human_relevance", "human_coherence", "human_consistency", "human_fluency", "human_sum"]):
    metric_train = "unieval_" + metric_target.split("_")[1]
    for repetitions, method_kwargs in [
        (100, dict(method="random")),
        (1, dict(method="metric_avg", metric=metric_train)),
        (1, dict(method="metric_var", metric=metric_train)),
        (1, dict(method="metric_cons", metric=metric_train)),
        (1, dict(method="diversity", metric="LM")),
        (5, dict(method="pyirt_diffdisc", metric=metric_train, retry_on_error=True)),
    ]:
        clu_all = []
        cor_all = []
        for _ in range(repetitions):
            data_new = subset2evaluate.select_subset.basic(data_old, **method_kwargs)
            clu, cor = subset2evaluate.evaluate.eval_clucor(data_new, data_old, metric=metric_target, props=PROPS)
            clu_all.append(clu)
            cor_all.append(cor)

        results[metric_target][method_kwargs["method"]] = (np.average(cor_all), np.average(clu_all))

# %%

# print column header and table def
with open("../figures_tex/49-everything_results_summeval.tex", "w") as f:
    print(
        r"\begin{tabular}{l"+
        "c"*(3+len(list(results.values())[0]))+
        "}\n"
        r"\toprule"
        "\n"
        r"\bf Dimension & ",
        r"\bf \hspace{-4mm}\#Models\hspace{-4mm} & \bf \#Items\hspace{-4mm} & ",
        file=f
    )
    for method, _ in list(results.values())[0].items():
        print(
            r"\bf " + f"{subset2evaluate.select_subset.methods.METHOD_NAMES[method]} & ",
            file=f
        )
    print(r"\\\midrule", file=f)

    for metric_target, methods in results.items():
        len_systems = len(data_old[0]["scores"].keys())
        len_items = len(data_old)
        
        # basic info in the first three columns
        metric_target = metric_target.split("_")[1].title()
        print(
            f"{metric_target} & ",
            f"{len_systems} & {len_items} &",
            file=f,
        )

        cor_best = f"{max(cor for cor, _ in methods.values()):.1%}"
        clu_best = f"{max(clu for _, clu in methods.values()):.2f}"

        for method, (cor, clu) in methods.items():
            cor = f"{cor:.1%}"
            clu = f"{clu:.2f}"

            # we are comparing rounded strings so no need to take decimals into account
            if cor == cor_best:
                cor = r"\textbf{" + cor + "}"
            if clu == clu_best:
                clu = r"\textbf{" + clu + "}"

            print(
                f"{cor}/{clu}".replace("100.0%", r"100%\phantom{.}").replace("%", r"\%"),
                " & ",
                file=f
            )
        print(r"\\", file=f)

    print(
        r"\bottomrule"
        r"\end{tabular}",
        file=f,
    )