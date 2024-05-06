import torch
import torch.utils
import lightning as L
import json


class IRTModel(L.LightningModule):
    def __init__(self, len_items, systems):
        super().__init__()

        # # make space for thresholded items
        # len_items = len_items*10

        # normally distribute at the beginning
        # discrimination
        self.param_a = torch.nn.Parameter(torch.randn(len_items))
        # difficulty
        self.param_b = torch.nn.Parameter(torch.randn(len_items))
        # mt perf
        self.param_theta = torch.nn.Parameter(torch.randn(len(systems)))
        self.systems = systems

        # self.loss_fn = torch.nn.L1Loss()
        self.loss_fn = torch.nn.MSELoss()
        # self.loss_fn = torch.nn.BCELoss()

    def forward(self, i_item, i_system):
        a = self.param_a[i_item]
        b = self.param_b[i_item]
        theta = self.param_theta[i_system]
        return 1 / (1 + torch.exp(-a * (theta - b)))

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        (i_item, i_system), y = batch
        y_hat = self.forward(i_item, i_system)

        # cast from f64 to f32
        y = y.float()
        loss = self.loss_fn(y_hat, y)

        # regularize
        # loss += sum([(p**2).sum() for p in self.param_theta])
        # loss += sum([(p**2).sum() for p in self.param_a])
        # loss += sum([(p**2).sum() for p in self.param_b])

        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        self.log("a_sign", torch.mean(torch.sign(self.param_a)))
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.1)

    def save_irt(self, filename):
        with open(filename, "w") as f:
            json.dump(
                obj={
                    "items": [
                        {"a": a, "b": b}
                        for a, b in zip(
                            self.param_a.tolist(),
                            self.param_b.tolist(),
                        )
                    ],
                    "systems": {
                        sys: sys_v
                        for sys, sys_v in zip(self.systems, self.param_theta.tolist())
                    },
                },
                fp=f,
            )
