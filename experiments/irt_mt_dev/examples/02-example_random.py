# python3 irt_mt_dev/peng/02-example_random.py

import irt_mt_dev.utils as utils
import random

data_full = utils.load_data_wmt()

for k in range(50, 500, 50):
    data_subset_random = random.Random(0).sample(data_full, k=k)
    print(k, utils.eval_order_accuracy(data_subset_random, data_full))