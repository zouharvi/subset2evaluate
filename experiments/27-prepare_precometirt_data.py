import pickle
import csv
import random
import argparse

args = argparse.ArgumentParser()
args.add_argument("--no-wmt23", action="store_true")
args = args.parse_args()

data_all = []

for i in range(9 if args.no_wmt23 else 0, 33):
    data_all += pickle.load(open(f"computed/irt_params/{i}.pkl", "rb"))

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

# prepare comet-compatible data
with open("../PreCOMET/data/csv/train_disc.csv", "w") as f:
    writer = csv.DictWriter(f, fieldnames=["src", "score"])
    writer.writeheader()
    writer.writerows(data_disc)

with open("../PreCOMET/data/csv/dev_disc.csv", "w") as f:
    writer = csv.DictWriter(f, fieldnames=["src", "score"])
    writer.writeheader()
    writer.writerows(random.Random(0).sample(data_disc, k=1000))

with open("../PreCOMET/data/csv/train_diff.csv", "w") as f:
    writer = csv.DictWriter(f, fieldnames=["src", "score"])
    writer.writeheader()
    writer.writerows(data_diff)

with open("../PreCOMET/data/csv/dev_diff.csv", "w") as f:
    writer = csv.DictWriter(f, fieldnames=["src", "score"])
    writer.writeheader()
    writer.writerows(random.Random(0).sample(data_diff, k=1000))

with open("../PreCOMET/data/csv/train_diffdisc.csv", "w") as f:
    writer = csv.DictWriter(f, fieldnames=["src", "score"])
    writer.writeheader()
    writer.writerows(data_diffdisc)

with open("../PreCOMET/data/csv/dev_diffdisc.csv", "w") as f:
    writer = csv.DictWriter(f, fieldnames=["src", "score"])
    writer.writeheader()
    writer.writerows(random.Random(0).sample(data_diffdisc, k=1000))