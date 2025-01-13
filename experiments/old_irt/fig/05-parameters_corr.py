import json
import scipy.stats

EPOCH = -1
data = {
    "human_0": json.load(open("computed/irt_wmt_4pl_s0_eall_human.json"))[EPOCH],
    "human_1": json.load(open("computed/irt_wmt_4pl_s1_eall_human.json"))[EPOCH],
    "metricx_0": json.load(open("computed/irt_wmt_4pl_s0_eall_metricx.json"))[EPOCH],
    "metricx_1": json.load(open("computed/irt_wmt_4pl_s1_eall_metricx.json"))[EPOCH],
    "bleu_0": json.load(open("computed/irt_wmt_4pl_s0_eall_bleu.json"))[EPOCH],
    "bleu_1": json.load(open("computed/irt_wmt_4pl_s1_eall_bleu.json"))[EPOCH],
}

for i1, (key1, val1) in enumerate(data.items()):
    print(f"{key1:>11}:", end=" ")
    for i2, (key2, val2) in enumerate(data.items()):
        if i2 > i1:
            continue
        # _val1 = [x["b"] for x in val1["items"]]
        # _val2 = [x["b"] for x in val2["items"]]
        corr = abs(scipy.stats.spearmanr(val1["theta"], val2["theta"])[0])
        if i1 == i2:
            print(f"{key1:>9}", end=" ")
        else:
            print(f"{corr:9.1%}", end=" ")

    print()
