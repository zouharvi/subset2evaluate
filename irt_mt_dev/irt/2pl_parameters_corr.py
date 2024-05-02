import json
import scipy.stats

data = {
    "score": json.load(open("computed/2pl_score.json")),
    "bleu": json.load(open("computed/2pl_BLEU.json")),
    "metricx": json.load(open("computed/2pl_MetricX-23-c.json")),
    "score_v1": json.load(open("computed/2pl_score_v1.json")),
    "bleu_v1": json.load(open("computed/2pl_BLEU_v1.json")),
    "metricx_v1": json.load(open("computed/2pl_MetricX-23-c_v1.json")),
}

print(" "*11, " ".join([f"{x:>7}" for x in data.keys()]))

for key1, val1 in data.items():
    print(f"{key1:>10}:", end=" ")
    for key2, val2 in data.items():
        corr = scipy.stats.spearmanr(val1["theta"], val2["theta"])[0]
        if corr == 1:
            print(f"{'### ':>7}", end=" ")
        else:
            print(f"{corr:7.0%}", end=" ")

    print()