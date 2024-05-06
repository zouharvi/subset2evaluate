import json
import scipy.stats

data = {
    "human": json.load(open("computed/irt_score.json")),
    "metricx": json.load(open("computed/irt_MetricX-23-c.json")),
    "bleu": json.load(open("computed/irt_BLEU.json")),
}

systems = list(list(data.values())[0]["systems"].keys())

print(" "*11, " ".join([f"{x:>7}" for x in data.keys()]))

for i1, (key1, val1) in enumerate(data.items()):
    print(f"{key1:>10}:", end=" ")
    for i2, (key2, val2) in enumerate(data.items()):
        # _val1 = [x["a"] for x in val1["items"]]
        # _val2 = [x["a"] for x in val2["items"]]
        if i2 > i1:
            continue
        _val1 = [val1["systems"][sys] for sys in systems]
        _val2 = [val2["systems"][sys] for sys in systems]
        corr = scipy.stats.spearmanr(_val1, _val2)[0]
        if corr == 1:
            print(f"{'######':>7}", end=" ")
        else:
            print(f"{corr:7.0%}", end=" ")

    print()