# %%
import collections
import subset2evaluate.utils as utils
import subset2evaluate.evaluate
import subset2evaluate.select_subset
import numpy as np
import tqdm
import itertools
import sacrebleu

data_old_all = list(utils.load_data_wmt_all(normalize=True).values())[:9]

cor_new_all = collections.defaultdict(list)
clu_new_all = collections.defaultdict(list)
metric_bleu = sacrebleu.metrics.BLEU(effective_order=True)


for data_old in tqdm.tqdm(data_old_all):
    def evaluate_balanced_domains(data_scored):
        data_aggregated = collections.defaultdict(list)
        for line in data_scored:
            data_aggregated[line["domain"]].append(line)
        
        # sort within each domain by utility
        data_aggregated = [
            sorted(domain, key=lambda x: x["subset2evaluate_utility"], reverse=True)
            for domain in data_aggregated.values()
        ]

        # interveawe the lists
        # this is a cheap trick that somewhat guarantees that data_new_flat[:k] has balanced domains
        # create a list of ABCDABCDABCDCDCDDDD
        data_new_flat = [doc for docs in itertools.zip_longest(*data_aggregated) for doc in docs]
        data_new_flat = [line for line in data_new_flat if line is not None]
        clu_new, cor_new = subset2evaluate.evaluate.eval_clucor(data_new_flat, data_old, metric="human")
        return np.average(clu_new), np.average(cor_new)
    
    for repetitions, method_kwargs in [
        (100, dict(method="random")),
        (1, dict(method="metric_avg", metric="MetricX-23")),
        (1, dict(method="metric_var", metric="MetricX-23")),
        (1, dict(method="diversity_bleu")),
        (1, dict(method="metric_alignment", metric="MetricX-23")),
        (5, dict(method="pyirt_diffdisc", metric="MetricX-23", model="4pl_score", epochs=1000, retry_on_error=True)),
    ]:
        data_y = subset2evaluate.select_subset.basic(data_old, **method_kwargs)
        clu_new, cor_new = evaluate_balanced_domains(data_y)
        cor_new_all[method_kwargs["method"]].append(cor_new)
        clu_new_all[method_kwargs["method"]].append(clu_new)

for method in cor_new_all.keys():
    cor_new = cor_new_all[method]
    clu_new = clu_new_all[method]
    print(method, f"COR: {np.average(cor_new):.1%} | CLU: {np.average(clu_new):.2f}")
