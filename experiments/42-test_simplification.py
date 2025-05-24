# %%

import collections
import csv
import numpy as np
import evaluate
import subset2evaluate.evaluate
import subset2evaluate.utils
import tqdm
import subset2evaluate

with open("../data/other/simpeval_past.csv", "r") as f:
    data_raw = list(csv.DictReader(f))


# %%
data = collections.defaultdict(list)
for line in data_raw:
    line_new = {
        "i": line["original_id"],
        "src": line["original"],
        "tgt": line["processed_generation"],
        "model": line["model"],
        "human": np.average([float(line[f"rating_{i}"]) for i in range(1, 5+1)]),
        "human_zscore": np.average([float(line[f"rating_{i}_z_score"]) for i in range(1, 5+1)]),
    }
    data[line_new["i"]].append(line_new)


data = [
    {
        "i": v[0]["i"],
        "src": v[0]["src"],
        "tgt": {
            x["model"]: x["tgt"]
            for x in v
        },
        "scores": {
            x["model"]: {
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
        "model": line["Input.model"],
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
            x["model"]: x["tgt"]
            for x in v
        },
        "scores": {
            x["model"]: x["scores"]
            for x in v
        }
    }
    for v in data.values()
]

sari = evaluate.load("sari")
for line in tqdm.tqdm(data):
    line["ref1"] = line["tgt"].pop("Human 1 Writing")
    line["scores"].pop("Human 1 Writing")
    line["ref2"] = line["tgt"].pop("Human 2 Writing")
    line["scores"].pop("Human 2 Writing")

    models = list(line["tgt"].keys())
    for model in models:
        line["scores"][model]["sari"] = sari.compute(sources=[line["src"]], predictions=[line["tgt"][model]], references=[[line["ref1"], line["ref2"]]])["sari"]

# %%
for line in data:
    for model, scores in line["scores"].items():
        scores["human_sum"] = scores["human_adequacy"] + scores["human_fluency"] + scores["human_simplicity"]
        scores["human_mul"] = scores["human_adequacy"] * scores["human_fluency"] * scores["human_simplicity"]
# %%

props = [0.25, 0.5, 0.75]

clus_new_sum = []
corss_new_sum = []
clus_new_mul = []
cors_new_mul = []
for _ in tqdm.tqdm(range(1000)):
    data_new = subset2evaluate.select_subset.basic(data, method="random")
    clu_new, cor_new = subset2evaluate.evaluate.eval_clu_cor(data_new, data, metric="human", props=props)
    clus_new_sum.append(np.average(clu_new))
    corss_new_sum.append(np.average(cor_new))
    clu_new, cor_new = subset2evaluate.evaluate.eval_clu_cor(data_new, data, metric="human_zscore", props=props)
    clus_new_mul.append(np.average(clu_new))
    cors_new_mul.append(np.average(cor_new))

print(f"sum | CLU: {np.average(clus_new_sum):.2f} | COR: {np.average(corss_new_sum):.1%}")
print(f"mul | CLU: {np.average(clus_new_mul):.2f} | COR: {np.average(cors_new_mul):.1%}")

# %%
data_new = subset2evaluate.select_subset.basic(data, method="diversity", metric="BLEU")
clu_new, cor_new = subset2evaluate.evaluate.eval_clu_cor(data_new, data, metric="human", props=props)
print(f"sum | CLU: {np.average(clu_new):.2f} | COR: {np.average(cor_new):.1%}")
clu_new, cor_new = subset2evaluate.evaluate.eval_clu_cor(data_new, data, metric="human_zscore", props=props)
print(f"mul | CLU: {np.average(clu_new):.2f} | COR: {np.average(cor_new):.1%}")

data_new = subset2evaluate.select_subset.basic(data, method="diversity", metric="ChrF")
clu_new, cor_new = subset2evaluate.evaluate.eval_clu_cor(data_new, data, metric="human", props=props)
print(f"sum | CLU: {np.average(clu_new):.2f} | COR: {np.average(cor_new):.1%}")
clu_new, cor_new = subset2evaluate.evaluate.eval_clu_cor(data_new, data, metric="human_zscore", props=props)
print(f"mul | CLU: {np.average(clu_new):.2f} | COR: {np.average(cor_new):.1%}")
# %%
data_new = subset2evaluate.select_subset.basic(data, method="metric_var", metric="sari")
clu_new, cor_new = subset2evaluate.evaluate.eval_clu_cor(data_new, data, metric="human_sum", props=props)
print(f"sum | CLU: {np.average(clu_new):.2f} | COR: {np.average(cor_new):.1%}")
clu_new, cor_new = subset2evaluate.evaluate.eval_clu_cor(data_new, data, metric="human_zscore", props=props)
print(f"mul | CLU: {np.average(clu_new):.2f} | COR: {np.average(cor_new):.1%}")

data_new = subset2evaluate.select_subset.basic(data, method="metric_avg", metric="sari")
clu_new, cor_new = subset2evaluate.evaluate.eval_clu_cor(data_new, data, metric="human_sum", props=props)
print(f"sum | CLU: {np.average(clu_new):.2f} | COR: {np.average(cor_new):.1%}")
clu_new, cor_new = subset2evaluate.evaluate.eval_clu_cor(data_new, data, metric="human_zscore", props=props)
print(f"mul | CLU: {np.average(clu_new):.2f} | COR: {np.average(cor_new):.1%}")
