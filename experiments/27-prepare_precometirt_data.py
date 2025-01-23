import pickle
import csv
import random
import argparse

args = argparse.ArgumentParser()
args.add_argument("--exclude", default="wmt23")
args = args.parse_args()

data_all = []

for i in range(53):
    data_name, data_val = pickle.load(open(f"computed/irt_params/{i}.pkl", "rb"))
    if data_name[0] == args.exclude:
        print("Skipping", data_name)
    else:
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

# prepare comet-compatible data
with open("../COMETsrc/data/csv/train_disc.csv", "w") as f:
    writer = csv.DictWriter(f, fieldnames=["src", "score"])
    writer.writeheader()
    writer.writerows(data_disc)

with open("../COMETsrc/data/csv/dev_disc.csv", "w") as f:
    writer = csv.DictWriter(f, fieldnames=["src", "score"])
    writer.writeheader()
    writer.writerows(random.Random(0).sample(data_disc, k=1000))

with open("../COMETsrc/data/csv/train_diff.csv", "w") as f:
    writer = csv.DictWriter(f, fieldnames=["src", "score"])
    writer.writeheader()
    writer.writerows(data_diff)

with open("../COMETsrc/data/csv/dev_diff.csv", "w") as f:
    writer = csv.DictWriter(f, fieldnames=["src", "score"])
    writer.writeheader()
    writer.writerows(random.Random(0).sample(data_diff, k=1000))

with open("../COMETsrc/data/csv/train_diffdisc.csv", "w") as f:
    writer = csv.DictWriter(f, fieldnames=["src", "score"])
    writer.writeheader()
    writer.writerows(data_diffdisc)

with open("../COMETsrc/data/csv/dev_diffdisc.csv", "w") as f:
    writer = csv.DictWriter(f, fieldnames=["src", "score"])
    writer.writeheader()
    writer.writerows(random.Random(0).sample(data_diffdisc, k=1000))
