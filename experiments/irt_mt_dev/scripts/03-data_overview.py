import irt_mt_dev.utils as utils

data_all = utils.load_data_wmt_all(min_segments=0)
data_all_filtered = utils.load_data_wmt_all()

print(
    sum([len(v) for k, v in data_all.items()]),
    sum([len(v) for k, v in data_all_filtered.items()]),
)
print(
    sum([len(v)*len(v[0]["scores"]) for k, v in data_all.items()]),
    sum([len(v)*len(v[0]["scores"]) for k, v in data_all_filtered.items()]),
)
print(
    len(data_all),
    len(data_all_filtered),
)
