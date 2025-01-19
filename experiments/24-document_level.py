# %%
import collections
import subset2evaluate.utils as utils
import subset2evaluate.evaluate
import subset2evaluate.select_subset
import numpy as np
import tqdm
import itertools
import sacrebleu

data_old_all = list(utils.load_data_wmt_all(normalize=True).values())[:9]
metric_bleu = sacrebleu.metrics.BLEU(effective_order=True)

# TODO: instead of reimplementing the utility functions here, use line["subset2evaluate_utility"]

def utility_diversity(line):
    score = -np.average([
        metric_bleu.sentence_score(
            text_a,
            [text_b],
        ).score
        for text_a, text_b in itertools.product(line["tgt"].values(), line["tgt"].values())
    ])
    return score

# %%
# aggregate scores


cor_new_all = collections.defaultdict(list)
clu_new_all = collections.defaultdict(list)

for data_old in tqdm.tqdm(data_old_all):
    data_old_aggregated = collections.defaultdict(list)
    for line in data_old:
        data_old_aggregated[line["doc"]].append(line)

    data_old_aggregated = [
        {
            "doc": doc,
            "i": [line["i"] for line in lines],
            "scores": {
                model: {
                    metric: np.average([line["scores"][model][metric] for line in lines])
                    for metric in lines[0]["scores"][model].keys()
                }
                for model in lines[0]["scores"].keys()
            },
            "tgt": {
                # concatenate all documents
                model: "\n".join([line["tgt"][model] for line in lines])
                for model in lines[0]["tgt"].keys()
            }
        }
        for doc, lines in data_old_aggregated.items()
    ]

    for method_kwargs in [
        dict(method="random"),
        dict(method="metric_avg"),
        dict(method="metric_var"),
        dict(method="diversity_bleu"),
        dict(method="pyirt_diffdisc", model="4pl_score", epochs=1000),
    ]:
        # run multiple times to average out the effect
        for _ in range(5 if method_kwargs["method"] == "pyirt_diffdisc" else 100 if method_kwargs["method"] == "random" else 1):
            data_new = subset2evaluate.select_subset.basic(
                data_old_aggregated,
                **method_kwargs,
                metric="MetricX-23-c",
                retry_on_error=True,
            )
            data_new_flat = [
                data_old[i]
                for doc in data_new
                for i in doc["i"]
            ]
            clu_new, cor_new = subset2evaluate.evaluate.eval_clucor(data_new_flat, data_old, metric="human")
            cor_new_all[method_kwargs["method"]].append(np.average(cor_new))
            clu_new_all[method_kwargs["method"]].append(np.average(clu_new))

print("Aggregate scores:")

for method, cor_new in cor_new_all.items():
    print(method, f"ACC: {np.average(cor_new):.1%}")

for method, clu_new in clu_new_all.items():
    print(method, f"CLU: {np.average(clu_new):.2f}")

# %%
# aggregate utilities

cor_new_all = collections.defaultdict(list)
clu_new_all = collections.defaultdict(list)

for data_old in tqdm.tqdm(data_old_all):
    def evaluate_aggregate_second(data_y):
        data_old_aggregated = collections.defaultdict(list)
        for line in data_old:
            data_old_aggregated[line["doc"]].append(line)
        data_old_aggregated = [
            {
                "doc": doc,
                "i": [line["i"] for line in lines],
                # average the utilities
                "score": np.average([data_y[line["i"]] for line in lines])
            }
            for doc, lines in data_old_aggregated.items()
        ]
        data_old_aggregated.sort(key=lambda x: x["score"], reverse=True)
        data_new_flat = [
            data_old[i]
            for doc in data_old_aggregated
            for i in doc["i"]
        ]
        clu_new, cor_new = subset2evaluate.evaluate.eval_clucor(data_new_flat, data_old, metric="human")
        return np.average(clu_new), np.average(cor_new)

    for _ in range(1):
        data_y = [np.var([line["scores"][model]["MetricX-23-c"] for model in line["scores"].keys()]) for line in data_old]
        clu_new, cor_new = evaluate_aggregate_second(data_y)
        cor_new_all["metric_var"].append(cor_new)
        clu_new_all["metric_var"].append(clu_new)

        data_y = [np.average([-line["scores"][model]["MetricX-23-c"] for model in line["scores"].keys()]) for line in data_old]
        clu_new, cor_new = evaluate_aggregate_second(data_y)
        cor_new_all["metric_avg"].append(cor_new)
        clu_new_all["metric_avg"].append(clu_new)

        data_y = [utility_diversity(line) for line in data_old]
        clu_new, cor_new = evaluate_aggregate_second(data_y)
        cor_new_all["diversity_bleu"].append(cor_new)
        clu_new_all["diversity_bleu"].append(clu_new)

    for _ in range(5):
        _, params = subset2evaluate.select_subset.basic(data_old, return_model=True, method="pyirt_diffdisc", model="4pl_score", metric="MetricX-23-c", epochs=1000, retry_on_error=True)
        data_y = [line_irt["diff"] * line_irt["disc"] for line_old, line_irt in zip(data_old, params["items"])]
        clu_new, cor_new = evaluate_aggregate_second(data_y)
        cor_new_all["pyirt_diffdisc"].append(cor_new)
        clu_new_all["pyirt_diffdisc"].append(clu_new)

    for _ in range(100):
        data_y = [np.random.random() for line in data_old]
        clu_new, cor_new = evaluate_aggregate_second(data_y)
        cor_new_all["random"].append(cor_new)
        clu_new_all["random"].append(clu_new)


print("Aggregate utility:")
for method, cor_new in cor_new_all.items():
    print(method, f"ACC: {np.average(cor_new):.1%}")

for method, clu_new in clu_new_all.items():
    print(method, f"CLU: {np.average(clu_new):.2f}")
