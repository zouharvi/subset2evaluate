import pickle
import csv
import random

data_all = []

for i in range(33):
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
        "score": line["irt"]["diff"]*line["irt"]["disc"],
    }
    for line in data_all
]

# # min-max normalize diff to [0, 1]
# diff_min = min([line["score"] for line in data_diff])
# diff_max = max([line["score"] for line in data_diff])
# for line in data_diff:
#     line["score"] = (line["score"] - diff_min) / (diff_max - diff_min)

# # min-max normalize disc to [0, 1]
# disc_min = min([line["score"] for line in data_disc])
# disc_max = max([line["score"] for line in data_disc])
# for line in data_disc:
#     line["score"] = (line["score"] - disc_min) / (disc_max - disc_min)


# prepare comet-compatible data
with open("../comet-src/data/csv/train_disc.csv", "w") as f:
    writer = csv.DictWriter(f, fieldnames=["src", "score"])
    writer.writeheader()
    writer.writerows(data_disc)

with open("../comet-src/data/csv/dev_disc.csv", "w") as f:
    writer = csv.DictWriter(f, fieldnames=["src", "score"])
    writer.writeheader()
    writer.writerows(random.Random(0).sample(data_disc, k=1000))

with open("../comet-src/data/csv/train_diff.csv", "w") as f:
    writer = csv.DictWriter(f, fieldnames=["src", "score"])
    writer.writeheader()
    writer.writerows(data_diff)

with open("../comet-src/data/csv/dev_diff.csv", "w") as f:
    writer = csv.DictWriter(f, fieldnames=["src", "score"])
    writer.writeheader()
    writer.writerows(random.Random(0).sample(data_diff, k=1000))

with open("../comet-src/data/csv/train_diffdisc.csv", "w") as f:
    writer = csv.DictWriter(f, fieldnames=["src", "score"])
    writer.writeheader()
    writer.writerows(data_diffdisc)

with open("../comet-src/data/csv/dev_diffdisc.csv", "w") as f:
    writer = csv.DictWriter(f, fieldnames=["src", "score"])
    writer.writeheader()
    writer.writerows(random.Random(0).sample(data_diffdisc, k=1000))


# sbatch_gpu "firstrun_diff" "comet-train --cfg configs/experimental/hypothesisless_model_diff.yaml"
# sbatch_gpu "firstrun_disc" "comet-train --cfg configs/experimental/hypothesisless_model_disc.yaml"
# sbatch_gpu_short "test" "python3 experiments/irt_mt_dev/subset_selection/17-quick_test.py"