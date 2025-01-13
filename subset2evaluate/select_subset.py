from typing import List, Any, Union, Tuple, Dict
import subset2evaluate.utils as utils
import subset2evaluate.methods as methods
import copy
import sys


def run_select_subset(
    data: Union[List, str],
    method: str,
    metric=None,
    return_model=False,
    load_model=None,
    retry_on_error=False,
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

    out_fn = lambda: method(
        data,
        metric=metric,
        return_model=return_model,
        load_model=load_model,
        **kwargs
    )

    # methods might mutate data, make sure we keep it clean
    data = copy.deepcopy(data)

    # pyirt does not handle divergence well and just crashes
    # on that occasion, let's just restart
    if retry_on_error:
        while True:
            try:
                out = out_fn()
            except Exception as e:
                print(e, file=sys.stderr)
                continue
    else:
        out = out_fn()

    if return_model:
        out, model = out

    out: List[Tuple[float, Dict]]

    # store utilities and sort from highest to lowest
    for score, item in zip(out, data):
        item["subset2evaluate_utility"] = score
    data.sort(key=lambda x: x["subset2evaluate_utility"], reverse=True)

    if return_model:
        return data, model
    else:
        return data


def main_cli():
    import argparse
    import json
    import ast

    args = argparse.ArgumentParser(
        description="""
        Select subset of data. The returned data is ordered by the method's utility in descending order (first is best).
        The segment utility is also stored in the 'subset2evaluate_utility' field of each item.
    """
    )
    args.add_argument(
        'data', type=str,
        default='wmt23/en-cs',
        help="Either descriptor of data, such as wmt22/en-de, or summeval, or path to JSON file with data."
    )
    args.add_argument(
        '--method', default="metric_var",
        choices=methods.METHODS.keys(),
        help="Subset selection method.",
    )
    args.add_argument(
        '--args',
        default='{"metric": "MetricX-23-c"}',
        help="Additional optional arguments for the method as a Python dictionary."
    )
    args = args.parse_args()

    data_new = run_select_subset(
        args.data,
        method=args.method,
        **ast.literal_eval(args.args)
    )

    for item in data_new:
        print(json.dumps(item, ensure_ascii=False))
