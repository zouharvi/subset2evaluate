import torch
import torch.utils
import lightning as L
import json

class IRTModelBase(L.LightningModule):
    def __init__(self, systems):
        super().__init__()

        # MT ability modelling is always the same across all models (scalar)
        self.param_theta = torch.nn.Parameter(torch.randn(len(systems)))

        self.systems = systems
        self.params_log = []
        self.loss_fn = torch.nn.MSELoss()

        self.clamp_feas = False

    def forward(self, i_item, i_system):
        disc = self.get_irt_params(i_item, "disc")
        diff = self.get_irt_params(i_item, "diff")
        feas = self.get_irt_params(i_item, "feas")
        theta = self.param_theta[i_system]

        return feas + (1 - feas) / (1 + torch.exp(-disc * (theta - diff)))

    def training_step(self, batch, batch_idx):
        # apply constraint
        if self.clamp_feas:
            self.param_feas.data = torch.clamp(self.param_feas, min=10e-3, max=1-10e-3)

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

        # TODO: monitor accuracy & clusters here to know when to stop

        self.log("val_loss_vs_constant", loss_pred-loss_const)

        self.params_log.append(self.pack_irt_params())

    def get_irt_params(self, i_item, name):
        raise NotImplementedError

    def pack_irt_params_items(self):
        raise NotImplementedError

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.1)
    
    def pack_irt_params(self):
        return {
            "items": self.pack_irt_params_items(),
            "systems": {
                sys: sys_v
                for sys, sys_v in zip(self.systems, self.param_theta.detach().tolist())
            },
        }

    def save_irt_params(self, filename):
        # save last parameters
        self.params_log.append(self.pack_irt_params())

        with open(filename, "w") as f:
            json.dump(obj=self.params_log, fp=f)