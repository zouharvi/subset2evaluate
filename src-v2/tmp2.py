import utils
import numpy as np
import random
data_old = utils.load_data()
data_old.sort(key=lambda line: max(line["score"].values())-min(line["score"].values()))

acc_total = []
for count_leave in range(10, 1100, 50):
    data_new = data_old[count_leave:]
    acc = utils.eval_data_pairs(data_new, data_old)
    print(f"{len(data_new)}/{len(data_old)}", f"{acc:.2%}")
    acc_total.append(acc)

print(np.average(acc_total))