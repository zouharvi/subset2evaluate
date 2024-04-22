import numpy as np
import irt_mt_dev.utils as utils
import torch
import torch.utils
import lightning as L
import argparse
from basic import IRTModel

data_wmt = utils.load_data(normalize=True)

args = argparse.ArgumentParser()
args.add_argument("--score", default="metric", choices=["human", "metric"])
args = args.parse_args()


systems = list(data_wmt[0]["score"].keys())
model = IRTModel(len(data_wmt), systems)

if args.score == "human":
    # TODO: better rescaling so that it's normally distributed?
    data_loader = [
        ((sent["i"], sys_i), sent["score"][sys]/110)
        for sent in data_wmt
        for sys_i, sys in enumerate(systems)
    ]
elif args.score == "metric":
    data_loader = [
        # special indexing
        ((sent["i"]*10+threshold_i, sys_i), (sent["metrics"][sys]["MetricX-23-c"]>=threshold)*1.0)
        for sent in data_wmt
        for sys_i, sys in enumerate(systems)
        # we have 0.1, 0.2, ... 1.0
        for threshold_i, threshold in enumerate(np.linspace(0, 1, 11)[1:])
    ]

data_loader = torch.utils.data.DataLoader(
    data_loader,
    batch_size=len(data_wmt),
    num_workers=24,
    shuffle=True,
    # fully move to GPU
    pin_memory=True,
    # don't kill workers because that's our bottleneck
    persistent_workers=True,
)

trainer = L.Trainer(max_epochs=100, log_every_n_steps=1)
trainer.fit(model=model, train_dataloaders=data_loader)
model.save_irt(f"computed/itr_{args.score}.json")
