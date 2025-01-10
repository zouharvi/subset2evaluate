import utils
import json

data_old = utils.load_data_wmt()

systems_irt = json.load(open("computed/irt_wmt_4pl_s0_eall_metricx.json"))[20]["theta"]
systems_wmt = utils.get_sys_absolute(data_old, metric="MetricX-23-c")
systems_irt = {
    k: v
    # hope that the ordering is the same
    for k, v in zip(systems_wmt.keys(), systems_irt)
}


print(utils.get_ord_accuracy(systems_irt, systems_wmt))