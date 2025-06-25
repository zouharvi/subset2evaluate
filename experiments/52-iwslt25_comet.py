# %%
import pickle
import subset2evaluate.utils

data_all = subset2evaluate.utils.load_data_iwslt(compute_automated_metrics=True)

with open("computed/iwslt25_comet.pkl", "wb") as f:
    pickle.dump(data_all, f)


"""
rsync -azP data/iwslt/*.jsonl euler:/cluster/work/sachan/vilem/subset2evaluate/data/iwslt/
sbatch_gpu_short "52-iwslt25_comet" "python3 experiments/52-iwslt25_comet.py"
scp euler:/cluster/work/sachan/vilem/subset2evaluate/computed/iwslt25_comet.pkl computed/iwslt25_comet.pkl
"""