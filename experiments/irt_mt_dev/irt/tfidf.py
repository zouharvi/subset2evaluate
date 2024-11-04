import torch
import torch.utils
from irt_mt_dev.irt.base import IRTModelBase
from sklearn.feature_extraction.text import TfidfVectorizer

class IRTModelTFIDF(IRTModelBase):
    def __init__(self, items, systems):
        super().__init__(systems=systems)

        encoder = TfidfVectorizer(max_features=768)
        self.text_src = torch.tensor(encoder.fit_transform([item["src"] for item in items]).toarray()).float().to("cuda")

        # normally distribute at the beginning
        # use the last parameter for bias
        self.param_disc = torch.nn.Parameter(torch.randn(768+1)).to("cuda")
        self.param_diff = torch.nn.Parameter(torch.randn(768+1)).to("cuda")
        self.param_feas = torch.nn.Parameter(torch.randn(768+1)).to("cuda")

        self.len_items = len(items)
    
    def get_irt_params(self, i_item, name):
        if name == "disc":
            return self.text_src[i_item,:] @ self.param_disc[:768] + self.param_disc[-1]
        elif name == "diff":
            return self.text_src[i_item,:] @ self.param_diff[:768] + self.param_diff[-1]
        elif name == "feas":
            return self.text_src[i_item,:] @ self.param_feas[:768] + self.param_feas[-1]
        
    def pack_irt_params_items(self):
        return [
            {
                "disc": self.get_irt_params(i_item, name="disc").detach().item(),
                "diff": self.get_irt_params(i_item, name="diff").detach().item(),
                "feas": self.get_irt_params(i_item, name="feas").detach().item(),
            }
            for i_item in range(self.len_items)
        ]