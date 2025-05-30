import pickle
import csv
import random
import argparse

args = argparse.ArgumentParser()
args.add_argument("split", choices=["test", "train"])
args = args.parse_args()

data_all = []

for i in range(49 if args.split == "train" else 9):
    data_name, data_val = pickle.load(open(f"computed/irt_params/{args.split}_{i}.pkl", "rb"))
    data_all += data_val

data_diff = [
    {
        "src": line["src"],
        "score": line["irt"]["diff"],
    }
    for line in data_all
]
data_disc = [
    {
        "src": line["src"],
        "score": line["irt"]["disc"],
    }
    for line in data_all
]
data_diffdisc = [
    {
        "src": line["src"],
        "score": line["irt"]["diff"] * line["irt"]["disc"],
    }
    for line in data_all
]


with open(f"../PreCOMET/data/csv/{args.split}_disc.csv", "w") as f:
    writer = csv.DictWriter(f, fieldnames=["src", "score"])
    writer.writeheader()
    writer.writerows(data_disc)

if args.split == "train":
    with open("../PreCOMET/data/csv/dev_disc.csv", "w") as f:
        writer = csv.DictWriter(f, fieldnames=["src", "score"])
        writer.writeheader()
        writer.writerows(random.Random(0).sample(data_disc, k=1000))

with open(f"../PreCOMET/data/csv/{args.split}_diff.csv", "w") as f:
    writer = csv.DictWriter(f, fieldnames=["src", "score"])
    writer.writeheader()
    writer.writerows(data_diff)

if args.split == "train":
    with open("../PreCOMET/data/csv/dev_diff.csv", "w") as f:
        writer = csv.DictWriter(f, fieldnames=["src", "score"])
        writer.writeheader()
        writer.writerows(random.Random(0).sample(data_diff, k=1000))

with open(f"../PreCOMET/data/csv/{args.split}_diffdisc.csv", "w") as f:
    writer = csv.DictWriter(f, fieldnames=["src", "score"])
    writer.writeheader()
    writer.writerows(data_diffdisc)

if args.split == "train":
    with open("../PreCOMET/data/csv/dev_diffdisc.csv", "w") as f:
        writer = csv.DictWriter(f, fieldnames=["src", "score"])
        writer.writeheader()
        writer.writerows(random.Random(0).sample(data_diffdisc, k=1000))


"""
python3 experiments/27-prepare_precometirt_data.py train
python3 experiments/27-prepare_precometirt_data.py test
"""