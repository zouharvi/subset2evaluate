import argparse
import json
import os
import irt_mt_dev.utils as utils
import methods
import copy

args = argparse.ArgumentParser()
args.add_argument('data', type=str, default='wmt23/en-cs')
args.add_argument('--method', default="var", choices=methods.METHODS.keys())
args.add_argument('--metric', default="MetricX-23")
args.add_argument('--model', default=None, choices=["scalar", "tfidf", "embd"])
args = args.parse_args()

if args.data.startswith("wmt"):
    data_year, data_lang = args.data.split("/")
    data = utils.load_data_wmt(year=data_year, langs=data_lang, normalize=True)
elif os.path.exists(args.data):
    data = [json.loads(x) for x in open(args.data, "r")]
else:
    raise Exception("Could not parse data")

method = args.method
if method not in methods.METHODS:
    raise Exception(f"Method {method} not found")
method = methods.METHODS[method]

# methods might mutate data, make sure we keep a copy
data_old = copy.deepcopy(data)
data_new = method(data, args)

for item in data_new:
    print(json.dumps(item, ensure_ascii=False))