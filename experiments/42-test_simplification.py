# %%

import collections
import csv
import json
import numpy as np
import evaluate
import subset2evaluate.evaluate
import subset2evaluate.utils

with open("../data/other/simpeval_past.csv", "r") as f:
    data_raw = list(csv.DictReader(f))


# %%
data = collections.defaultdict(list)
for line in data_raw:
    line_new = {
        "i": line["original_id"],
        "src": line["original"],
        "tgt": line["processed_generation"],
        "system": line["system"],
        "human": np.average([float(line[f"rating_{i}"]) for i in range(1, 5+1)]),
        "human_zscore": np.average([float(line[f"rating_{i}_z_score"]) for i in range(1, 5+1)]),
    }
    data[line_new["i"]].append(line_new)


data = [
    {
        "i": v[0]["i"],
        "src": v[0]["src"],
        "tgt": {
            x["system"]: x["tgt"]
            for x in v
        },
        "scores": {
            x["system"]: {
                "human": x["human"],
                "human_zscore": x["human_zscore"],
            }
            for x in v
        }
    }
    for v in data.values()
]

# %%


with open("../data/other/simpDA_2022.csv", "r") as f:
    data_raw = list(csv.DictReader(f))

data = collections.defaultdict(list)
for line in data_raw:
    line_new = {
        "i": line["Input.id"],
        "src": line["Input.original"],
        "tgt": line["Input.simplified"],
        "system": line["Input.system"],
        "scores": {
            "human_adequacy": float(line["Answer.adequacy"]),
            "human_fluency": float(line["Answer.fluency"]),
            "human_simplicity": float(line["Answer.simplicity"]),
        },
    }
    data[line_new["i"]].append(line_new)


data = [
    {
        "i": v[0]["i"],
        "src": v[0]["src"],
        "tgt": {
            x["system"]: x["tgt"]
            for x in v
        },
        "scores": {
            x["system"]: x["scores"]
            for x in v
        }
    }
    for v in data.values()
]

import tqdm
sari = evaluate.load("sari")
for line in tqdm.tqdm(data):
    line["ref1"] = line["tgt"].pop("Human 1 Writing")
    line["scores"].pop("Human 1 Writing")
    line["ref2"] = line["tgt"].pop("Human 2 Writing")
    line["scores"].pop("Human 2 Writing")

    systems = list(line["tgt"].keys())
    for sys in systems:
        line["scores"][sys]["sari"] = sari.compute(sources=[line["src"]], predictions=[line["tgt"][sys]], references=[[line["ref1"], line["ref2"]]])["sari"]

# %%
for line in data:
    for sys, scores in line["scores"].items():
        scores["human_sum"] = scores["human_adequacy"] + scores["human_fluency"] + scores["human_simplicity"]
        scores["human_mul"] = scores["human_adequacy"] * scores["human_fluency"] * scores["human_simplicity"]
# %%
import subset2evaluate
import tqdm

props = [0.25, 0.5, 0.75]

clus_new_sum = []
accs_new_sum = []
clus_new_mul = []
accs_new_mul = []
for _ in tqdm.tqdm(range(1000)):
    data_new = subset2evaluate.select_subset.basic(data, method="random")
    clu_new, acc_new = subset2evaluate.evaluate.eval_cluacc(data_new, data, metric="human", props=props)
    clus_new_sum.append(np.average(clu_new))
    accs_new_sum.append(np.average(acc_new))
    clu_new, acc_new = subset2evaluate.evaluate.eval_cluacc(data_new, data, metric="human_zscore", props=props)
    clus_new_mul.append(np.average(clu_new))
    accs_new_mul.append(np.average(acc_new))

print(f"sum | CLU: {np.average(clus_new_sum):.2f} | ACC: {np.average(accs_new_sum):.1%}")
print(f"mul | CLU: {np.average(clus_new_mul):.2f} | ACC: {np.average(accs_new_mul):.1%}")

# %%
data_new = subset2evaluate.select_subset.basic(data, method="diversity_bleu")
clu_new, acc_new = subset2evaluate.evaluate.eval_cluacc(data_new, data, metric="human", props=props)
print(f"sum | CLU: {np.average(clu_new):.2f} | ACC: {np.average(acc_new):.1%}")
clu_new, acc_new = subset2evaluate.evaluate.eval_cluacc(data_new, data, metric="human_zscore", props=props)
print(f"mul | CLU: {np.average(clu_new):.2f} | ACC: {np.average(acc_new):.1%}")

data_new = subset2evaluate.select_subset.basic(data, method="diversity_chrf")
clu_new, acc_new = subset2evaluate.evaluate.eval_cluacc(data_new, data, metric="human", props=props)
print(f"sum | CLU: {np.average(clu_new):.2f} | ACC: {np.average(acc_new):.1%}")
clu_new, acc_new = subset2evaluate.evaluate.eval_cluacc(data_new, data, metric="human_zscore", props=props)
print(f"mul | CLU: {np.average(clu_new):.2f} | ACC: {np.average(acc_new):.1%}")
# %%
data_new = subset2evaluate.select_subset.basic(data, method="metric_var", metric="sari")
clu_new, acc_new = subset2evaluate.evaluate.eval_cluacc(data_new, data, metric="human_sum", props=props)
print(f"sum | CLU: {np.average(clu_new):.2f} | ACC: {np.average(acc_new):.1%}")
clu_new, acc_new = subset2evaluate.evaluate.eval_cluacc(data_new, data, metric="human_zscore", props=props)
print(f"mul | CLU: {np.average(clu_new):.2f} | ACC: {np.average(acc_new):.1%}")

data_new = subset2evaluate.select_subset.basic(data, method="metric_avg", metric="sari")
clu_new, acc_new = subset2evaluate.evaluate.eval_cluacc(data_new, data, metric="human_sum", props=props)
print(f"sum | CLU: {np.average(clu_new):.2f} | ACC: {np.average(acc_new):.1%}")
clu_new, acc_new = subset2evaluate.evaluate.eval_cluacc(data_new, data, metric="human_zscore", props=props)
print(f"mul | CLU: {np.average(clu_new):.2f} | ACC: {np.average(acc_new):.1%}")