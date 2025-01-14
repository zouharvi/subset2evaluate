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

acc_new_all = collections.defaultdict(list)
clu_new_all = collections.defaultdict(list)
metric_bleu = sacrebleu.metrics.BLEU(effective_order=True)

# TODO: instead of reimplementing the utility functions here, use line["subset2evaluate_utility"]

def utility_diversity(line):
    return -np.average([
        metric_bleu.sentence_score(
            text_a,
            [text_b],
        ).score
        for text_a, text_b in itertools.product(line["tgt"].values(), line["tgt"].values())
    ])


for data_old in tqdm.tqdm(data_old_all):
    def evaluate_balanced_domains(data_y):
        data_aggregated = collections.defaultdict(list)
        for line in data_old:
            data_aggregated[line["domain"]].append({**line, "score": data_y[line["i"]]})
        # sort within each domain by score
        data_aggregated = [sorted(domain, key=lambda x: x["score"], reverse=True) for domain in data_aggregated.values()]
        # interveawe the lists
        # this is a cheap trick that somewhat guarantees that data_new_flat[:k] has balanced domains
        # create a list of ABCDABCDABCDCDCDDDD
        data_new_flat = [doc for docs in itertools.zip_longest(*data_aggregated) for doc in docs]
        data_new_flat = [line for line in data_new_flat if line is not None]
        clu_new, acc_new = subset2evaluate.evaluate.eval_cluacc(data_new_flat, data_old, metric="human")
        return np.average(clu_new), np.average(acc_new)

    for _ in range(1):
        data_y = [np.var([line["scores"][sys]["MetricX-23-c"] for sys in line["scores"].keys()]) for line in data_old]
        clu_new, acc_new = evaluate_balanced_domains(data_y)
        acc_new_all["metric_var"].append(acc_new)
        clu_new_all["metric_var"].append(clu_new)

        data_y = [np.average([-line["scores"][sys]["MetricX-23-c"] for sys in line["scores"].keys()]) for line in data_old]
        clu_new, acc_new = evaluate_balanced_domains(data_y)
        acc_new_all["metric_avg"].append(acc_new)
        clu_new_all["metric_avg"].append(clu_new)

        data_y = [utility_diversity(line) for line in data_old]
        clu_new, acc_new = evaluate_balanced_domains(data_y)
        acc_new_all["diversity_bleu"].append(acc_new)
        clu_new_all["diversity_bleu"].append(clu_new)

    for _ in range(5):
        _, params = subset2evaluate.select_subset.run_select_subset(data_old, return_model=True, method="pyirt_diffdisc", model="4pl_score", metric="MetricX-23-c", epochs=1000, retry_on_error=True)
        data_y = [line_irt["diff"] * line_irt["disc"] for line_irt in params["items"]]
        clu_new, acc_new = evaluate_balanced_domains(data_y)
        acc_new_all["pyirt_diffdisc"].append(acc_new)
        clu_new_all["pyirt_diffdisc"].append(clu_new)

    for _ in range(100):
        data_y = [np.random.random() for line in data_old]
        clu_new, acc_new = evaluate_balanced_domains(data_y)
        acc_new_all["random"].append(acc_new)
        clu_new_all["random"].append(clu_new)


print("Aggregate utility:")
for method, acc_new in acc_new_all.items():
    print(method, f"ACC: {np.average(acc_new):.1%}")

for method, clu_new in clu_new_all.items():
    print(method, f"CLU: {np.average(clu_new):.2f}")
