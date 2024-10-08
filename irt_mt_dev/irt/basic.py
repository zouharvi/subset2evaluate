import torch
import torch.utils
import lightning as L
import json
import copy

class IRTModel(L.LightningModule):
    def __init__(self, len_items, systems):
        super().__init__()

        # normally distribute at the beginning
        # discrimination
        self.param_disc = torch.nn.Parameter(torch.randn(len_items))
        # difficulty
        self.param_diff = torch.nn.Parameter(torch.randn(len_items))
        # feasability
        self.param_feas = torch.nn.Parameter(torch.randn(len_items))
        # MT ability
        self.param_theta = torch.nn.Parameter(torch.randn(len(systems)))
        self.systems = systems
        self.len_items = len_items

        self.params_log = []

        # self.loss_fn = torch.nn.L1Loss()
        self.loss_fn = torch.nn.MSELoss()
        # self.loss_fn = torch.nn.BCELoss()

    def forward(self, i_item, i_system):
        disc = self.param_disc[i_item]
        diff = self.param_diff[i_item]
        feas = self.param_feas[i_item]
        theta = self.param_theta[i_system]
        return feas + (1 - feas) / (1 + torch.exp(-disc * (theta - diff)))

    def training_step(self, batch, batch_idx):
        # apply constraint
        self.param_feas.data = torch.clamp(self.param_feas, min=0, max=1)
        
        # training_step defines the train loop.
        # it is independent of forward
        (i_item, i_system), y = batch
        y_hat = self.forward(i_item, i_system)

        # cast from f64 to f32
        y = y.float()
        loss = self.loss_fn(y_hat, y)

        # regularize
        # loss += torch.pow(self.param_a, 2).sum()/100


        self.log("train_loss", loss)
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

        self.params_log.append({
            "theta": self.param_theta.clone().detach().tolist(),
            "disc": self.param_disc.clone().detach().tolist(),
            "diff": self.param_diff.clone().detach().tolist(),
            "feas": self.param_feas.clone().detach().tolist(),
        })

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)

    def save_irt(self, filename):
        # save last parameters
        self.params_log.append({
            "theta": self.param_theta.clone().detach().tolist(),
            "disc": self.param_disc.clone().detach().tolist(),
            "diff": self.param_diff.clone().detach().tolist(),
            "feas": self.param_feas.clone().detach().tolist(),
        })

        with open(filename, "w") as f:
            json.dump(
                obj=[
                    {
                        "items": [
                            {"disc": disc, "diff": diff, "feas": feas}
                            for disc, diff, feas in zip(
                                params["disc"],
                                params["diff"],
                                params["feas"],
                            )
                        ],
                        "systems": {
                            sys: sys_v
                            for sys, sys_v in zip(self.systems, params["theta"])
                        },
                    } for params in self.params_log
                ],
                fp=f,
            )

    def load_irt(self, f_or_fname):
        raise Exception("WARNING: Loading is untested")
        if type(f_or_fname) == str:
            f_or_fname = open(f_or_fname, "r")
        
        data = json.load(f_or_fname)

        # make sure the ordering is the same
        assert set(self.systems) == set(data["systems"].keys())
        for sys_i, sys in enumerate(self.systems):
            self.param_theta[sys_i] = data["systems"][sys]

        for i in range(self.len_items):
            # need to inverse because we're saving softplused version
            self.param_disc[i] = data["items"][i]["a"]
            self.param_diff[i] = data["items"][i]["b"]

        f_or_fname.close()