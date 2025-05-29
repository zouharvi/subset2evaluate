# %%

import subset2evaluate.evaluate
import subset2evaluate.utils
import collections
import tqdm

# %%

data_old_all = subset2evaluate.utils.load_data_wmt_test()
results_all = collections.defaultdict(lambda: collections.defaultdict(list))

for data_old_name, data_old in tqdm.tqdm(list(data_old_all.items())):
    clusters_old = subset2evaluate.evaluate.compute_clusters(data_old)
    for repetitions, method_kwargs in [
        (100, dict(method="random")),
        (1, dict(method="metric_avg", metric="MetricX-23-c")),
        (1, dict(method="metric_var", metric="MetricX-23-c")),
        (1, dict(method="metric_cons", metric="MetricX-23-c")),
        (1, dict(method="diversity", metric="LM")),
        (5, dict(method="pyirt_diffdisc", metric="MetricX-23-c", retry_on_error=True)),
    ]:
        for _ in range(repetitions):
            data_new = subset2evaluate.select_subset.basic(data_old, **method_kwargs)
            for prop in subset2evaluate.utils.PROPS:
                k = int(len(data_old) * prop)
                results_all[method_kwargs["method"]]["spa"].append(
                    subset2evaluate.evaluate.eval_subset_spa(data_new[:k], data_old)
                )
                results_all[method_kwargs["method"]]["pairwise_accuracy"].append(
                    subset2evaluate.evaluate.eval_subset_pairwise_accuracy(data_new[:k], data_old)
                )
                results_all[method_kwargs["method"]]["kendall"].append(
                    subset2evaluate.evaluate.eval_subset_correlation(data_new[:k], data_old, correlation="kendall")
                )
                results_all[method_kwargs["method"]]["spearman"].append(
                    subset2evaluate.evaluate.eval_subset_correlation(data_new[:k], data_old, correlation="spearman")
                )
                results_all[method_kwargs["method"]]["pearson"].append(
                    subset2evaluate.evaluate.eval_subset_correlation(data_new[:k], data_old, correlation="pearson")
                )
                results_all[method_kwargs["method"]]["clusters"].append(
                    subset2evaluate.evaluate.eval_subset_clusters(data_new[:k])
                )
                results_all[method_kwargs["method"]]["top1"].append(
                    subset2evaluate.evaluate.eval_subset_top(data_new[:k], data_old)
                )
                results_all[method_kwargs["method"]]["clusters_top1"].append(
                    subset2evaluate.evaluate.eval_subset_clusters_top(data_new[:k], data_old)
                )
                results_all[method_kwargs["method"]]["error_absolute"].append(
                    subset2evaluate.evaluate.eval_subset_error(data_new[:k], data_old, error="absolute")
                )
                results_all[method_kwargs["method"]]["error_root_squared"].append(
                    subset2evaluate.evaluate.eval_subset_error(data_new[:k], data_old, error="root_squared")
                )

# %%
import numpy as np

METAEVAL_TO_NAME = {
    "spa": "Soft pairwise accuracy",
    "pairwise_accuracy": "Pairwise accuracy",
    "pearson": "Pearson correlation",
    "spearman": "Spearman correlation",
    "kendall": "Kendall\\textsubscript{b} correlation",
    "top1": "Top-1 match",
    "clusters": "Cluster count",
    "clusters_top1": "Top-1 cluster match",    
    "error_absolute": r"Mean average error $\downarrow$",
    "error_root_squared": r"Mean root squared error $\downarrow$",
}
METHOD_TO_NAME = {
    "random": "Random",
    "metric_avg": "MetricAvg",
    "metric_var": "MetricVar",
    "metric_cons": "MetricCons",
    "diversity": "Diversity",
    "pyirt_diffdisc": "DiffDisc",
}


with open("../figures_tex/49-other_meta_evaluation.tex", "w") as f:
    print(
        r"\begin{tabular}{l" + " c" * len(METHOD_TO_NAME) + r"} \\ \toprule",
        file=f,
    )
    print(
        r"\bf Meta-evaluation",
        *[f"\\bf {method_name}" for method_name in METHOD_TO_NAME.values()],
        sep=" & ",
        end="\\\\ \n \\midrule \n",
        file=f,
    )
    for metaeval, metaeval_name in METAEVAL_TO_NAME.items():
        # find best in each row
        best_method = max(
            METHOD_TO_NAME.keys(),
            key=lambda method: (
                (-1 if metaeval in {"error_absolute", "error_root_squared"} else 1) *
                np.average(results_all[method][metaeval])
            ),
        )

        print(
            metaeval_name,
            *[
                (r"\bf " if method == best_method else "") + (
                    f"{np.average(results_all[method][metaeval]):.2f}"
                    if metaeval in {"clusters"} else
                    f"{np.average(results_all[method][metaeval]):.3f}"
                    if metaeval in {"error_absolute", "error_root_squared"} else
                    f"{np.average(results_all[method][metaeval]):.1%}".replace("%", r"\%")
                )
                for method in METHOD_TO_NAME
            ],
            sep=" & ",
            end="\\\\ \n",
            file=f,
        )
    print(r"\bottomrule \end{tabular}", file=f)