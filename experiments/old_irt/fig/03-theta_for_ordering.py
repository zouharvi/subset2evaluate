import subset2evaluate.utils as utils
import json
import subset2evaluate

data_old = utils.load_data_wmt()

models_irt = json.load(open("computed/irt_wmt_4pl_s0_eall_metricx.json"))[20]["theta"]
models_wmt = subset2evaluate.evaluate.get_model_absolute(data_old, metric="MetricX-23-c")
models_irt = {
    k: v
    # hope that the ordering is the same
    for k, v in zip(models_wmt.keys(), models_irt)
}


print(utils.eval_order_accuracy(models_irt, models_wmt))
