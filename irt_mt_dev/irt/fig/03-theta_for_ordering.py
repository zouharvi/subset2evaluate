import irt_mt_dev.utils as utils
import json

data_old = utils.load_data()

systems_irt = json.load(open("computed/irt_metric.json"))["systems"]
systems_wmt = utils.get_sys_absolute(data_old, metric="score")

print(utils.get_ord_accuracy(systems_irt, systems_wmt))