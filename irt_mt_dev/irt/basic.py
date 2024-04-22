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
        # guessing
        self.param_c = torch.nn.Parameter(torch.randn(len_items))
        # mt perf
        self.param_theta = torch.nn.Parameter(torch.randn(len(systems)))
        self.systems = systems

    def forward(self, i_item, i_system):
        return self.param_c[i_item] + (1 - self.param_c[i_item]) / (
            1
            + torch.exp(
                -self.param_a[i_item] *
                (self.param_theta[i_system] - self.param_b[i_item])
            )
        )

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        (i_item, i_system), y = batch
        y_hat = self.forward(i_item, i_system)

        # cast from f64 to f32
        y = y.float()
        loss = torch.nn.functional.l1_loss(y_hat, y)

        # force zero mean and unit variance
        theta_std, theta_mean = torch.std_mean(self.param_theta)
        # force std to 1
        loss += torch.abs(theta_std-1)
        # force mean to 0
        loss += torch.abs(theta_mean)

        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        self.log("theta_mean", theta_mean)
        self.log("theta_std", theta_std)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def save_irt(self, filename):
        with open(filename, "w") as f:
            json.dump(
                obj={
                    "items": [
                        {"a": a, "b": b, "c": c}
                        for a, b, c in zip(
                            self.param_a.tolist(),
                            self.param_b.tolist(),
                            self.param_c.tolist(),
                        )
                    ],
                    "systems": {
                        sys: sys_v
                        for sys, sys_v in zip(self.systems, self.param_theta.tolist())
                    },
                },
                fp=f,
            )
