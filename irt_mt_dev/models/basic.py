import utils
import torch
import torch.utils
import lightning as L
import json

data_old = utils.load_data()


# define the LightningModule
class IRTModel(L.LightningModule):
    def __init__(self, len_item, len_mt):
        super().__init__()

        # normally distribute at the beginning
        # discrimination
        self.param_a = torch.nn.Parameter(torch.randn(len_item))
        # difficulty
        self.param_b = torch.nn.Parameter(torch.randn(len_item))
        # guessing
        self.param_c = torch.nn.Parameter(torch.randn(len_item))
        # mt perf
        self.param_mt = torch.nn.Parameter(torch.randn(len_mt))

    def forward(self, i_item, i_mt):
        return self.param_c[i_item] + (1 - self.param_c[i_item]) / (
            1
            + torch.exp(
                -self.param_a[i_item] * (self.param_mt[i_mt] - self.param_b[i_item])
            )
        )

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        (i_item, i_mt), y = batch
        y_hat = self.forward(i_item, i_mt)

        # cast from f64 to f32
        y = y.float()
        loss = torch.nn.functional.mse_loss(y_hat, y)

        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def save(self):
        with open("computed/itr.json", "w") as f:
            json.dump(
                obj={
                    "items": list(
                        zip(
                            self.param_a.tolist(),
                            self.param_b.tolist(),
                            self.param_c.tolist(),
                        )
                    ),
                    "mts": self.param_mt.tolist(),
                },
                fp=f,
            )


model = IRTModel(len(data_old), len(data_old[0]["score"].keys()))
# TODO: better rescaling so that it's normally distributed?
# HUMAN
# rescale everything from 0 to 1 for now
# data_loader = torch.utils.data.DataLoader(
#     [
#         ((sent["i"], mt_i), mt_v / 100)
#         for sent in data_old
#         for mt_i, mt_v in enumerate(sent["score"].values())
#     ],
#     batch_size=256,
#     num_workers=24,
# )

# TODO: better rescaling so that it's normally distributed?
# METRIC
# rescale everything from 0 to 1 for now
systems = list(data_old[0]["score"].keys())
data_loader = torch.utils.data.DataLoader(
    [
        ((sent["i"], sys_i), sent["metrics"][sys]["MetricX-23-c"]/10+1)
        for sent in data_old
        for sys_i, sys in enumerate(systems)
    ],
    batch_size=256,
    num_workers=24,
)

trainer = L.Trainer(max_epochs=300)
trainer.fit(model=model, train_dataloaders=data_loader)
model.save()