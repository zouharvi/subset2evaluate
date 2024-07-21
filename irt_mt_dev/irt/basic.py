import torch
import torch.utils
import lightning as L
import json

def softplus_inverse(x):
    return x + torch.log(-torch.expm1(-x))


class IRTModel(L.LightningModule):
    def __init__(self, len_items, systems):
        super().__init__()

        # normally distribute at the beginning
        # discrimination
        self.param_a = torch.nn.Parameter(torch.randn(len_items))
        # difficulty
        self.param_b = torch.nn.Parameter(torch.randn(len_items))
        # feasability
        self.param_c = torch.nn.Parameter(torch.randn(len_items))
        # mt ability
        self.param_theta = torch.nn.Parameter(torch.randn(len(systems)))
        self.systems = systems
        self.len_items = len_items

        # self.loss_fn = torch.nn.L1Loss()
        self.loss_fn = torch.nn.MSELoss()
        # self.loss_fn = torch.nn.BCELoss()

    def forward(self, i_item, i_system):
        a = self.param_a[i_item]
        b = self.param_b[i_item]
        c = self.param_c[i_item]
        theta = self.param_theta[i_system]
        return torch.sigmoid(c) / (1 + torch.exp(-a * (theta - b)))

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

        self.log("train_loss", loss)
        self.log("a_sign", torch.mean(torch.sign(self.param_a)))
        return loss
    

    def validation_step(self, batch, batch_idx):
        (x_items, x_systems), y_true = batch
        y_pred = self.forward(x_items, x_systems)

        loss_pred = torch.nn.functional.l1_loss(y_pred, y_true)
        loss_const = torch.nn.functional.l1_loss(
            torch.full_like(y_true, fill_value=torch.mean(y_true)),
            y_true
        )

        self.log("val_loss_vs_constant", loss_pred-loss_const)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)

    def save_irt(self, filename):
        with open(filename, "w") as f:
            json.dump(
                obj={
                    "items": [
                        {"a": a, "b": b, "c": c}
                        for a, b, c in zip(
                            self.param_a.tolist(),
                            self.param_b.tolist(),
                            torch.sigmoid(self.param_c).tolist(),
                        )
                    ],
                    "systems": {
                        sys: sys_v
                        for sys, sys_v in zip(self.systems, self.param_theta.tolist())
                    },
                },
                fp=f,
            )

    def load_irt(self, f_or_fname):
        print("WARNING: Loading is untested")
        if type(f_or_fname) == str:
            f_or_fname = open(f_or_fname, "r")
        
        data = json.load(f_or_fname)

        # make sure the ordering is the same
        assert set(self.systems) == set(data["systems"].keys())
        for sys_i, sys in enumerate(self.systems):
            self.param_theta[sys_i] = data["systems"][sys]

        for i in range(self.len_items):
            # need to inverse because we're saving softplused version
            self.param_a[i] = data["items"][i]["a"]
            self.param_b[i] = data["items"][i]["b"]

        f_or_fname.close()