import numpy as np
import irt_mt_dev.utils as utils
import torch
import torch.utils
import lightning as L
import argparse
from basic import IRTModel

data_wmt = utils.load_data(normalize=True)
# data_wmt = utils.get_nice_subset(data_wmt, target_size=100, step_size=50, metric="score")
# print(len(data_wmt), "nice lines")

args = argparse.ArgumentParser()
args.add_argument("--score", default="metric", choices=["human", "metric"])
args = args.parse_args()


systems = list(data_wmt[0]["score"].keys())
model = IRTModel(len(data_wmt), systems)

if args.score == "human":
    data_loader = [
        ((sent_i, sys_i), sent["score"][sys]>0.9)
        for sent_i, sent in enumerate(data_wmt)
        for sys_i, sys in enumerate(systems)
    ]
elif args.score == "metric":
    data_loader = [
        ((sent_i, sys_i), sent["metrics"][sys]["MetricX-23-c"])
        for sent_i, sent in enumerate(data_wmt)
        for sys_i, sys in enumerate(systems)
    ]
    _median = np.median([y for x, y in data_loader])
    data_loader = [
        (x, 1*(y>_median))
        for x, y in data_loader
    ]

data_loader = torch.utils.data.DataLoader(
    data_loader,
    batch_size=len(data_wmt)*len(systems),
    num_workers=24,
    shuffle=True,
    # fully move to GPU
    pin_memory=True,
    # don't kill workers because that's our bottleneck
    persistent_workers=True,
)

trainer = L.Trainer(max_epochs=1000, log_every_n_steps=1)
trainer.fit(model=model, train_dataloaders=data_loader)
model.save_irt(f"computed/itr_{args.score}.json")
