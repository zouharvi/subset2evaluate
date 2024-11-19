from typing import List

def load_data(data: List | str):
    import irt_mt_dev.utils as utils
    import os
    import json

    if type(data) is list:
        pass
    elif data.startswith("wmt"):
        data_year, data_lang = data.split("/")
        data = utils.load_data_wmt(year=data_year, langs=data_lang, normalize=True)
    elif os.path.exists(data):
        data = [json.loads(x) for x in open(data, "r")]
    else:
        raise Exception("Could not parse data")
    
    return data


# TODO: migrate important utils function from irt_mt_dev to here