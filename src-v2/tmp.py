import utils
import numpy as np
data_old = utils.load_data()

results = []
for line in data_old:
    scores = np.array(list(line["score"].values()))
    val_median = np.median(scores)
    for sys_v in scores:
        print(f"{sys_v:>3.0f}", end=" ")
    
    scores = [x for x in scores if x != 0]
    val_abs_dev = np.abs(scores-val_median)
    print(f"| {np.std(scores):.0f}", end="")
    print(f"| {np.average(val_abs_dev):.0f}", end="")
    print(f"| {np.max(val_abs_dev):.0f}")

    results.append({"scores": list(scores), "max-min": max(scores)-min(scores)})

results.sort(key=lambda x: x["max-min"])
for r in results[-10:]:
    print(r)