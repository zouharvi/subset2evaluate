import numpy as np
import torch
import torch.utils
import lightning as L
import json
import irt_mt_dev.utils as utils
import subset2evaluate.evaluate

class IRTModelBase(L.LightningModule):
    def __init__(self, systems, **kwargs):
        super().__init__()

        self.data_old = kwargs["data_old"]

        # MT ability modelling is always the same across all models (scalar)
        self.param_theta = torch.nn.Parameter(torch.randn(len(systems)))

        self.systems = systems
        self.results_log = []
        self.params_log = []
        self.loss_fn = torch.nn.MSELoss()

        self.clamp_feas = False

        
        def fn_information_content(item, system_thetas):
            information = 0
            for theta in system_thetas.values():
                prob = utils.pred_irt(
                    theta,
                    item
                )
                information += prob*(1-prob)*(item["disc"]**2)
            return information

        self.fn_utility = {
            "diff": lambda item, *args: -item["diff"],
            "disc": lambda item, *args: -item["disc"],
            "feas": lambda item, *args: item["feas"],
            "fisher_information_content": fn_information_content,
        }[kwargs["fn_utility"]]

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
        # we ignore the inputs and use the current parameters which are aligned with the full data

        # TODO: monitor accuracy & clusters here to know when to stop
        data_irt = self.pack_irt_params()
        items_joint = list(zip(self.data_old, data_irt["items"]))
        items_joint.sort(
            key=lambda x: self.fn_utility(x[1], data_irt["systems"]),
            reverse=True
        )

        data_new = [x[0] for x in items_joint]
        
        clu_new_metric, acc_new_metric = subset2evaluate.evaluate.run_evaluate_topk(
            self.data_old,
            data_new,
            # TODO: look at human and how it changes
            # TODO: set this dynamically
            metric="MetricX-23"
        )

        clu_new_human, acc_new_human = subset2evaluate.evaluate.run_evaluate_topk(
            self.data_old,
            data_new,
            # TODO: set this dynamically
            metric="human"
        )
        
        # print(
        # f"Metric CLU: {np.average(clu_new_metric):.2f} | ACC: {np.average(acc_new_metric):.2%} | ",
        # f"Human  CLU: {np.average(clu_new_human):.2f} | ACC: {np.average(acc_new_human):.2%}",
        # )

        self.log("cluster_count_metric", np.average(clu_new_metric))
        self.log("subset_consistency_accuracy_metric", np.average(acc_new_metric))
        self.log("cluster_count_human", np.average(clu_new_human))
        self.log("subset_consistency_accuracy_human", np.average(acc_new_human))

        self.results_log.append({
            "cluster_count_metric": np.average(clu_new_metric),
            "subset_consistency_accuracy_metric": np.average(acc_new_metric),
            "cluster_count_human": np.average(clu_new_human),
            "subset_consistency_accuracy_human": np.average(acc_new_human),
        })
        self.params_log.append(self.pack_irt_params())

    def get_irt_params(self, i_item, name):
        raise NotImplementedError

    def pack_irt_params_items(self):
        raise NotImplementedError

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.05)
    
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
