import argparse
import json
import os
import numpy as np
import irt_mt_dev.utils as utils

args = argparse.ArgumentParser()
args.add_argument('data_old', type=str, default='wmt23/en-cs')
args.add_argument('data_new', type=str, default='wmt23/en-cs')
args.add_argument('--metric', type=str, default='human')
args = args.parse_args()

if args.data_old.startswith("wmt"):
    data_year, data_lang = args.data_old.split("/")
    data_old = utils.load_data(year=data_year, langs=data_lang, normalize=True)
elif os.path.exists(args.data_old):
    data_old = [json.loads(x) for x in open(args.data_old, "r")]
else:
    raise Exception("Could not parse data")


if args.data_new.startswith("wmt"):
    data_year, data_lang = args.data_new.split("/")
    data_new = utils.load_data(year=data_year, langs=data_lang, normalize=True)
elif os.path.exists(args.data_new):
    data_new = [json.loads(x) for x in open(args.data_new, "r")]
else:
    raise Exception("Could not parse data")

clu_old = []
clu_new = []
acc_new = []
for prop in utils.PROPS:
    k = int(len(data_old)*prop)
    clu_old.append(utils.eval_system_clusters(data_old, metric=args.metric))
    clu_new.append(utils.eval_system_clusters(data_new[:k], metric=args.metric))
    acc_new.append(utils.eval_order_accuracy(data_new[:k], data_old, metric=args.metric))
print(f"Clusters (old->new): {np.average(clu_old):.3f} -> {np.average(clu_new):.3f}")
print(f"Accuracy (new): {np.average(acc_new):.2%}")