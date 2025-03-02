from typing import List, Any, Union, Tuple, Dict
import subset2evaluate.utils as utils
import subset2evaluate.methods as methods
import copy
import sys
import numbers


def basic(
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
                break
            except Exception as e:
                print(e, file=sys.stderr)
                continue
    else:
        out = out_fn()

    # make sure that we always return a tuple if return_model is True
    if return_model:
        if len(out) == 2 and isinstance(out[0], list) and all([isinstance(x, numbers.Number) for x in out[0]]):
            out, model = out
        else:
            out, model = out, None

    out: List[Tuple[float, Dict]]

    # store utilities and sort from highest to lowest
    for score, item in zip(out, data):
        item["subset2evaluate_utility"] = score
    data.sort(key=lambda x: x["subset2evaluate_utility"], reverse=True)

    if return_model:
        return data, model
    else:
        return data


def costaware(
    data: Union[List, str],
    budget: float,
    time_limit: float = 60,
    **kwargs
) -> List:
    """
    Requires items to have "subset2evaluate_utility" field, which commonly means that the data need to run
    through the subset2evaluate.select_subset.basic(..) method first.

    Each item also needs to have positive "cost" field.
    """
    import scipy.optimize
    import numpy as np

    assert all(["subset2evaluate_utility" in x for x in data]), "Items need to have 'subset2evaluate_utility' field."
    assert all(["cost" in x for x in data]), "Items need to have 'cost' field."

    # make sure utility is always positive
    data_new_utility = np.array([x["subset2evaluate_utility"] for x in data])
    min_data = min(data_new_utility)
    max_data = max(data_new_utility)
    data_new_utility = (data_new_utility - min_data) / (max_data - min_data) / 2
    data_new_utility += 0.1

    # simulate random cost
    # import random
    # costs = np.array([x["cost"] for x in data])
    # random.shuffle(costs)

    opt = scipy.optimize.milp(
        # minimize negative utility
        c=-data_new_utility,
        bounds=scipy.optimize.Bounds(0, 1),
        constraints=scipy.optimize.LinearConstraint(
            A=[line["cost"] for line in data],
            lb=0,
            ub=budget,
        ),
        # has to be integer
        integrality=np.full_like(data, 1),
        options=dict(
            time_limit=time_limit,
        )
    )

    # greedily fill budget
    top_k = list(np.argsort(opt.x))
    data_new = []
    while len(top_k) != 0:
        new_line = data[top_k.pop()]
        if sum([line["cost"] for line in data_new+[new_line]]) >= budget:
            break
        data_new.append(new_line)

    # data_new = [line for x, line in zip(opt.x, data) if x == 1.0]

    return data_new


def main_cli():
    import argparse
    import json
    import ast

    args = argparse.ArgumentParser(
        description="""
            Select subset of data. The returned data is ordered by the method's utility in descending order (first is best).
            The item utility is also stored in the 'subset2evaluate_utility' field of each item.
        """
    )
    args.add_argument(
        'data', type=str,
        default='wmt23/en-cs',
        help="Either descriptor of data, such as wmt22/en-de, or summeval, or path to JSON file with data."
    )
    args.add_argument(
        '--method', default="metric_var",
        choices=[k for k in methods.METHODS.keys() if not k.startswith("local_")],
        help="Subset selection method.",
    )
    args.add_argument(
        '--args',
        default='{"metric": "MetricX-23-c"}',
        help="Additional optional arguments for the method as a Python dictionary."
    )
    args = args.parse_args()

    data_new = basic(
        args.data,
        method=args.method,
        **ast.literal_eval(args.args)
    )

    for item in data_new:
        print(json.dumps(item, ensure_ascii=False))
