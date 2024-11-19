import subset2evaluate.select_subset
import subset2evaluate.evaluate

data_new = subset2evaluate.select_subset.run_select_subset("wmt23/en-cs", method="avg", metric="MetricX-23")

(clu_old, clu_new), acc_new = subset2evaluate.evaluate.run_evaluate_topk("wmt23/en-cs", data_new, metric="MetricX-23")
print(acc_new)