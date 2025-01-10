# %%

from typing import Dict, List
import subset2evaluate.evaluate
import subset2evaluate.select_subset


def load_summeval():
    from datasets import load_dataset
    import collections
    data_raw = load_dataset("KnutJaegersberg/summeval_pairs")["train"]

    data_by_id = collections.defaultdict(list)
    for line in data_raw:
        data_by_id[line["id"]].append(line)

    def avg_human_annotations(expert_annotations: List[Dict[str, float]]) -> Dict[str, float]:
        scores = collections.defaultdict(list)
        for line in expert_annotations:
            for k, v in line.items():
                scores[k].append(v)
        return {"human_" + k: sum(v) / len(v) for k, v in scores.items()}


    data = []
    for i, v in data_by_id.items():
        # "coherence": 2, "consistency": 1, "fluency": 4, "relevance": 2 
        data.append({
            "i": i,
            "src": None,
            "ref": None,
            "tgt": {line["model_id"]: line["decoded"] for line in v},
            "scores": {
                # rouge is nested for some reason
                line["model_id"]: (
                    line["metric_scores_1"] | line["metric_scores_1"]["rouge"] | avg_human_annotations(line["expert_annotations"])
                )
                for line in v
            },
        })

    # remove rouge from scores
    data = [
        {
            **line,
            "scores": {
                sys: {
                    metric: score for metric, score in metrics.items()
                    if metric != "rouge"
                }
                for sys, metrics in line["scores"].items()
            }
        }
        for line in data
    ]

    return list(data)

data_old = load_summeval()

# %%

import subset2evaluate.select_subset
import subset2evaluate.evaluate
import numpy as np

acc_all = []
clu_all = []
for _ in range(100):
    data_new = subset2evaluate.select_subset.run_select_subset(data_old, method="random")
    clu_new, acc_new = subset2evaluate.evaluate.run_evaluate_topk(data_old, data_new, metric="human_relevance")
    acc_all.append(acc_new)
    clu_all.append(clu_new)
print(f"ACC: {np.average(acc_all):.2%} | CLU: {np.average(clu_all):.2f}")

# %%

for metric in data_old[0]["scores"]["M11"].keys():
    data_new = subset2evaluate.select_subset.run_select_subset(data_old, method="avg", metric=metric)
    clu_new, acc_new = subset2evaluate.evaluate.run_evaluate_topk(data_old, data_new, metric="human_coherence")
    print(f"{metric:>20}", f"ACC: {np.average(acc_new):.2%} | CLU: {np.average(clu_new):.2f}")


# %%
data_new = subset2evaluate.select_subset.run_select_subset(data_old, method="diversity")
clu_new, acc_new = subset2evaluate.evaluate.run_evaluate_topk(data_old, data_new, metric="human_relevance")
print(f"ACC: {np.average(acc_new):.2%} | CLU: {np.average(clu_new):.2f}")