# %%

import subset2evaluate
import subset2evaluate.evaluate
import subset2evaluate.utils
import subset2evaluate.select_subset
import numpy as np
import collections
import tqdm

data_old_all = subset2evaluate.utils.load_data('wmt/all')

results = collections.defaultdict(dict)

for data_old_name, data_old in tqdm.tqdm(list(data_old_all.items())):
    # skip data with no metrics (just human)
    if len(list(data_old[0]["scores"].values())[0]) == 1:
        continue
    for repetitions, method_kwargs in [
        (10, dict(method="random")),
        (1, dict(method="metric_avg", metric=None)),
        (1, dict(method="metric_var", metric=None)),
        (1, dict(method="metric_cons", metric=None)),
        (1, dict(method="diversity", metric="LM")),
        (1, dict(method="pyirt_diffdisc", metric=None, retry_on_error=True)),
    ]:

        # if metric is None, find best automatically
        if "metric" in method_kwargs and method_kwargs["metric"] is None:
            metrics = subset2evaluate.evaluate.eval_metrics_correlations(data_old, metric_target="human")
            # output dict is sorted
            method_kwargs["metric"] = list(metrics.keys())[0]

        clu_all = []
        cor_all = []
        for _ in range(repetitions):
            data_new = subset2evaluate.select_subset.basic(data_old, **method_kwargs)
            clu, cor = subset2evaluate.evaluate.eval_clucor(data_new, data_old)
            clu_all.append(clu)
            cor_all.append(cor)

        results[data_old_name][method_kwargs["method"]] = (np.average(cor_all), np.average(clu_all))

# %%
# sort by WMT year
results = collections.OrderedDict(sorted(results.items(), key=lambda x: x[0][0], reverse=True))
results_years_len = collections.Counter(data_old_name[0].replace(".news", ".tedtalks") for data_old_name in results.keys())

# print column header and table def
with open("../figures_tex/48-everything_results_wmt.tex", "w") as f:
    print(
        r"\begin{tabular}{ll"+
        "c"*(3+len(list(results.values())[0]))+
        "}\n"
        r"\toprule"
        "\n"
        r"\multicolumn{2}{l}{\bf Dataset} & ",
        r"\bf \hspace{-4mm}\#Models\hspace{-4mm} & \bf \#Items\hspace{-4mm} & ",
        file=f
    )
    prev_wmt = None
    for method, _ in list(results.values())[0].items():
        print(
            r"\bf " + f"{subset2evaluate.select_subset.methods.METHOD_NAMES[method]} & ",
            file=f
        )
    print(r"\\\midrule", file=f)

    for (wmt_year, wmt_langs), methods in results.items():
        extra_line = False
        # print separator and dataset name
        if wmt_year != prev_wmt and ".news" not in wmt_year:
            extra_line = True
            # print WMT year rotated
            print(
                r"\parbox[t]{2mm}{\multirow{" + 
                    str(results_years_len[wmt_year.replace(".tedtalks", "").replace(".news", "")]) + 
                    r"}{*}{\rotatebox[origin=c]{90}{" +
                    wmt_year.replace(".tedtalks", "").replace(".news", "").upper() +
                r"}}}",
                file=f,
            )
        prev_wmt = wmt_year

        len_systems = len(data_old_all[(wmt_year, wmt_langs)][0]["scores"].keys())
        len_items = len(data_old_all[(wmt_year, wmt_langs)])
        
        # basic info in the first three columns
        wmt_langs = wmt_langs.split("-")
        wmt_langs = f"\\tto{{{wmt_langs[0].title()}}}{{{wmt_langs[1].title()}}}"
        print(
            f"& {wmt_langs} & ",
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
        if not extra_line:
            print(r"\\", file=f)
        else:
            print(r"\\[0.5em]", file=f)

    print(
        r"\bottomrule"
        r"\end{tabular}",
        file=f,
    )