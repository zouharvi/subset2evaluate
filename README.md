# subset2evaluate

> TODO

```bash
bash scripts/01-get_data.sh

# install the subset2evaluate package
cd ../subset2evaluate
pip3 install -e .
```

# Dependencies and Requirements

TODO, add checks that our py-irt and comet-src is installed

# Example 1

TODO process this

```
# This is how to set up the data loader and everything
# Run all the scripts from the top-level directory:
# pip3 install -e .
# bash 01-get_data.sh
# This will register the project as a python package and downloads the data
# Then, in Python, you can do:

import utils
data = utils.load_data_wmt()

# In most cases we want to normalize the data such that values are in [0, 1]
data = utils.load_data_wmt(normalize=True)

# There are other options that load a different language pair from a different year but the default should be fine for now
# The `data` is a list of items
# Each item has:
# - `i` - the index of the item
# - `domain` - the domain of the sentence
# - `src` - the source sentence
# - `ref` - the reference sentence
# - `tgt` - mapping from systems to their translations
# - `scores` - mapping from systems to score names to numbers
#            - for example data[0]["scores"]["ONLINE-Y"]["human"] is the human judgement for ONLINE-Y system for the first item
#            - for example data[5]["scores"]["NLLB_GReedy"]["MetricX-23"] is MetricX-23 score for NLLB_GReedy system for the sixth item
```

# Example 2

TODO process this

```
import utils
import random

data_full = utils.load_data_wmt()

for k in range(50, 500, 50):
    data_subset_random = random.Random(0).sample(data_full, k=k)
    print(k, utils.eval_subset_accuracy(data_subset_random, data_full))
```