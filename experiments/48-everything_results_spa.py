# %%

import subset2evaluate
import subset2evaluate.evaluate
import subset2evaluate.utils
import subset2evaluate.select_subset
import numpy as np
import collections
import tqdm

data_old_all_wmt = subset2evaluate.utils.load_data('wmt/all')
results_wmt = collections.defaultdict(dict)

for data_old_name, data_old in tqdm.tqdm(list(data_old_all_wmt.items())):
    # skip data with no metrics (just human)
    if len(list(data_old[0]["scores"].values())[0]) == 1:
        continue
    for repetitions, method_kwargs in [
        (100, dict(method="random")),
        (1, dict(method="metric_avg", metric=None)),
        (1, dict(method="metric_var", metric=None)),
        (1, dict(method="metric_cons", metric=None)),
        (1, dict(method="diversity", metric="LM")),
        (5, dict(method="pyirt_diffdisc", metric=None, retry_on_error=True)),
    ]:

        # if metric is None, find best automatically
        if "metric" in method_kwargs and method_kwargs["metric"] is None:
            metrics = subset2evaluate.evaluate.eval_metrics_correlations(data_old, metric_target="human")
            # output dict is sorted
            method_kwargs["metric"] = list(metrics.keys())[0]

        spa_all = []
        for _ in range(repetitions):
            data_new = subset2evaluate.select_subset.basic(data_old, **method_kwargs)
            spa = subset2evaluate.evaluate.eval_spa(data_new, data_old)
            spa_all.append(spa)

        results_wmt[data_old_name][method_kwargs["method"]] = np.average(spa_all)

# %%

data_old_summeval = subset2evaluate.utils.load_data_summeval(load_extra=True)
PROPS = np.geomspace(0.25, 0.75, 10)

results_summeval = collections.defaultdict(dict)

for metric_target in tqdm.tqdm(["human_relevance", "human_coherence", "human_consistency", "human_fluency", "human_sum", "human_mul"]):
    metric_train = "gpt_" + metric_target.split("_")[1]
    for repetitions, method_kwargs in [
        (100, dict(method="random")),
        (1, dict(method="metric_avg", metric=metric_train)),
        (1, dict(method="metric_var", metric=metric_train)),
        (1, dict(method="metric_cons", metric=metric_train)),
        (1, dict(method="diversity", metric="LM")),
        (5, dict(method="pyirt_diffdisc", metric=metric_train, retry_on_error=True)),
    ]:
        spa_all = []
        for _ in range(repetitions): 
            data_new = subset2evaluate.select_subset.basic(data_old_summeval, **method_kwargs)
            spa = subset2evaluate.evaluate.eval_spa(data_new, data_old_summeval, metric=metric_target, props=PROPS)
            spa_all.append(spa)

        results_summeval[metric_target][method_kwargs["method"]] = np.average(spa_all)
# %%
# sort by WMT year
results_wmt = collections.OrderedDict(sorted(results_wmt.items(), key=lambda x: x[0][0], reverse=True))
results_years_len = collections.Counter(data_old_name[0].replace(".news", "").replace(".tedtalks", "") for data_old_name in results_wmt.keys())

# print column header and table def
with open("../figures_tex/48-everything_results_spa.tex", "w") as f:
    print(
        r"\begin{tabular}{ll"+
        "c"*(3+len(list(results_wmt.values())[0]))+
        "}\n"
        r"\toprule"
        "\n"
        r"\multicolumn{2}{l}{\bf Dataset} & ",
        r"\bf \hspace{-4mm}\#Models\hspace{-4mm} & \bf \#Items\hspace{-4mm} & ",
        file=f
    )
    prev_wmt = None
    for method, _ in list(results_wmt.values())[0].items():
        print(
            r"\bf " + f"{subset2evaluate.select_subset.methods.METHOD_NAMES[method]} & ",
            file=f
        )
    print(r"\\\midrule", file=f)

    for (wmt_year, wmt_langs), methods in results_wmt.items():
        extra_line = False
        # print separator and dataset name
        if wmt_year != prev_wmt and ".news" not in wmt_year:
            extra_line = True
            # print WMT year rotated
            print(
                r"\parbox[t]{2mm}{\multirow{" + 
                    str(results_years_len[wmt_year.replace(".tedtalks", "").replace(".news", "")]) + 
                    r"}{*}{\rotatebox[origin=c]{90}{" +
                    wmt_year.replace(".tedtalks", "").replace(".news", "").upper().replace("WMT19", r"WMT19\hspace{3mm}") +
                r"}}}",
                file=f,
            )
        prev_wmt = wmt_year

        len_systems = len(data_old_all_wmt[(wmt_year, wmt_langs)][0]["scores"].keys())
        len_items = len(data_old_all_wmt[(wmt_year, wmt_langs)])
        
        # basic info in the first three columns
        wmt_langs = wmt_langs.split("-")
        wmt_langs = f"\\tto{{{wmt_langs[0].title()}}}{{{wmt_langs[1].title()}}}"
        print(
            f"& {wmt_langs} & ",
            f"{len_systems} & {len_items} &",
            file=f,
        )


        spa_best = f"{max(list(methods.values())):.1%}"

        for method, spa in methods.items():
            spa = f"{spa:.1%}"

            # we are comparing rounded strings so no need to take decimals into account
            if spa == spa_best:
                spa = r"\textbf{" + spa + "}"

            print(
                f"{spa}".replace("100.0%", r"100%\phantom{.}").replace("%", r"\%"),
                " & ",
                file=f
            )
        if not extra_line:
            print(r"\\", file=f)
        else:
            print(r"\\[0.5em]", file=f)


    # SUMMEVAL
    print(r"\\[-0.2em]", file=f)
    print(
        r"\parbox[t]{2mm}{\multirow{6}{*}{\rotatebox[origin=c]{90}{SummEval}}}",
        file=f,
    )
    for metric_target, methods in results_summeval.items():
        len_systems = len(data_old[0]["scores"].keys())
        len_items = len(data_old)
        
        # basic info in the first three columns
        metric_target = metric_target.split("_")[1].title()
        print(
            f"& {metric_target} & ",
            f"{len_systems} & {len_items} &",
            file=f,
        )

        cor_best = f"{max(list(methods.values())):.1%}"

        for method, spa in methods.items():
            spa = f"{spa:.1%}"

            # we are comparing rounded strings so no need to take decimals into account
            if spa == cor_best:
                spa = r"\textbf{" + spa + "}"

            print(
                f"{spa}".replace("100.0%", r"100%\phantom{.}").replace("%", r"\%"),
                " & ",
                file=f
            )
        print(r"\\", file=f)

    print(
        r"\bottomrule"
        r"\end{tabular}",
        file=f,
    )