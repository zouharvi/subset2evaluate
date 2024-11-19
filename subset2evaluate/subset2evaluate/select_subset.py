import json
import os
import irt_mt_dev.utils as utils
import methods
import copy

def run_select_subset(data, method, metric=None, model=None):
    if data.startswith("wmt"):
        data_year, data_lang = data.split("/")
        data = utils.load_data_wmt(year=data_year, langs=data_lang, normalize=True)
    elif os.path.exists(data):
        data = [json.loads(x) for x in open(data, "r")]
    else:
        raise Exception("Could not parse data")

    if method not in methods.METHODS:
        raise Exception(f"Method {method} not found")
    method = methods.METHODS[method]

    # methods might mutate data, make sure we keep it clean
    data_new = method(copy.deepcopy(data), model=model, metric=metric)

    return data_new


if __name__ == "__main__":
    import argparse

    args = argparse.ArgumentParser()
    args.add_argument('data', type=str, default='wmt23/en-cs')
    args.add_argument('--method', default="var", choices=methods.METHODS.keys())
    args.add_argument('--metric', default="MetricX-23")
    args.add_argument('--model', default=None, choices=["scalar", "tfidf", "embd"])
    args = args.parse_args()

    data_new = run_select_subset(args.data, args.method, args.metric, args.model)

    for item in data_new:
        print(json.dumps(item, ensure_ascii=False))