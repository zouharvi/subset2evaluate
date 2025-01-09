# %%
import collections
import subset2evaluate.select_subset
import subset2evaluate.evaluate
import subset2evaluate.utils
import random
import tqdm
import numpy as np
import irt_mt_dev.utils as utils
import os
os.chdir("/home/vilda/irt-mt-dev")

random.seed(0)

data_old_all = list(utils.load_data_wmt_all(normalize=True).items())[:9]

# number of systems in each language pair
print([len(data_old[0]["scores"]) for data_old_name, data_old in data_old_all])
SUBSET_SIZE = [2, 4]

# %%
accs_all = collections.defaultdict(lambda: collections.defaultdict(list))
clus_all = collections.defaultdict(lambda: collections.defaultdict(list))
cache_data = {}

for subset_size in tqdm.tqdm(SUBSET_SIZE, desc="Subset size"):
    # run multiple times
    for data_old_name, data_old in tqdm.tqdm(data_old_all, desc="Language pair"):
        for _ in tqdm.tqdm(range(5), desc="Repetition"):
            systems = list(data_old[0]["scores"].keys())
            systems_true = random.sample(systems, k=8)
            systems_artificial = random.sample(sorted(set(systems)-set(systems_true)), k=subset_size)
            
            data_old_true = [
                {
                    **line,
                    "scores": {
                        sys: v
                        for sys, v in line["scores"].items()
                        if sys in systems_true
                    },
                    "tgt": {
                        sys: v
                        for sys, v in line["tgt"].items()
                        if sys in systems_true
                    }
                }
                for line in data_old
            ]
            data_old_i_to_line = {line["i"]: line for line in data_old_true}
            data_old_artificial = [
                {
                    **line,
                    "scores": {
                        sys: v
                        for sys, v in line["scores"].items()
                        if sys in systems_artificial
                    },
                    "tgt": {
                        sys: v
                        for sys, v in line["tgt"].items()
                        if sys in systems_artificial
                    }
                }
                for line in data_old
            ]

            # we dropped some systems but we can recover them with the same ordering from data_old
            for cache, method_kwargs in [
                (False, dict(method="random")),
                (True,  dict(method="precomet_var")),
                (True,  dict(method="precomet_avg")),
                (True,  dict(method="precomet_div")),
                (True,  dict(method="precomet_diff_precomet_disc")),
                (False, dict(method="var", metric="MetricX-23-c")),
                (False, dict(method="avg", metric="MetricX-23-c")),
                (False, dict(method="output_text_var")),
                (False, dict(method="pyirt_diffdisc", model="4pl_score", metric="MetricX-23-c")),
            ]:
                if cache and (method_kwargs["method"], data_old_name) in cache_data:
                    data_new = cache_data[(method_kwargs["method"], data_old_name)]
                else:
                    data_new = subset2evaluate.select_subset.run_select_subset(data_old_artificial, **method_kwargs, retry_on_error=True)
                    if cache:
                        cache_data[(method_kwargs["method"], data_old_name)] = data_new

                clu_new, acc_new = subset2evaluate.evaluate.run_evaluate_topk(data_old_true, [
                    data_old_i_to_line[line["i"]]
                    for line in data_new
                ], metric="human")
                clus_all[method_kwargs["method"]][subset_size].append(clu_new)
                accs_all[method_kwargs["method"]][subset_size].append(acc_new)

# %%
def method_formatter(method):
    DICT = {
        "random": "Random",
        "precomet_var": r"PreCOMET\textsuperscript{var}",
        "precomet_avg": r"PreCOMET\textsuperscript{avg}",
        "precomet_div": r"PreCOMET\textsuperscript{div}",
        "var": "MetricX var.",
        "avg": "MetricX avg.",
        "output_text_var": "Diversity",
        "pyirt_diffdisc": r"IRT diff.$\times$disc.",
        "precomet_diff_precomet_disc": r"PreCOMET\textsuperscript{diff.$\times$disc}",
        "precomet_diffdisc": r"PreCOMET\textsuperscript{diff.$\times$disc} (direct)",
    }
    if method in DICT:
        return DICT[method]
    else:
        return method

# ACCs
for method, subset_kv in accs_all.items():
    print(method_formatter(method), end=" & ")
    if method == "random" or method.startswith("precomet"):
        print(
            r"\multicolumn{2}{c}{",
            "------",
            f"{np.average([np.average(subset_kv[subset_size]) for subset_size in SUBSET_SIZE]):.2%}".replace("%", r"\%"),
            "------",
            "}",
            end=" \\\\\n"
        )
    else:
        print(
            *[f"{np.average(subset_kv[subset_size]):.2%}".replace("%", r"\%") for subset_size in SUBSET_SIZE],
            sep=" & ",
            end=" \\\\\n"
        )

# CLUs
print()
for method, subset_kv in clus_all.items():
    print(method_formatter(method), end=" & ")
    if method == "random" or method.startswith("precomet"):
        print(
            r"\multicolumn{2}{c}{",
            "------",
            f"{np.average([np.average(subset_kv[subset_size]) for subset_size in SUBSET_SIZE]):.2f}" ,
            "------",
            "}",
            end=" \\\\\n"
        )
    else:
        print(
            *[f"{np.average(subset_kv[subset_size]):.2f}" for subset_size in SUBSET_SIZE],
            sep=" & ",
            end=" \\\\\n"
        )
