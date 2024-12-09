import json
from typing import List, Any, Union, Tuple
import subset2evaluate.utils as utils
import subset2evaluate.methods as methods
import copy

def run_select_subset(
        data : List | str,
        method : str,
        metric=None,
        model=None,
        return_model=False,
        load_model=None,
        **kwargs
    ) -> Union[List, Tuple[List, Any]]:
    """
    Returs list of ordered data and possibly also the model, if return_model=True. Not all methods support this though.
    """
    # both list or descriptor is fine
    data = utils.load_data(data)

    if method not in methods.METHODS:
        raise Exception(f"Method {method} not found")
    method = methods.METHODS[method]

    # methods might mutate data, make sure we keep it clean
    data = copy.deepcopy(data)
    return method(
        data,
        model=model,
        metric=metric,
        return_model=return_model,
        load_model=load_model,
        **kwargs
    )

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