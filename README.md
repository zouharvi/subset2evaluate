# <img src="misc/logo.svg" height="25em"> subset2evaluate

Package to select informative samples to human-evaluate for NLG tasks such as machine translation or summarization.

> TODO abstract


# Example for Machine Translation

Install the package and download WMT data:
```bash
pip3 install subset2evaluate
bash experiments/01-get_wmt_data.sh
```

Then in Python we compute the baseline:
```python
import subset2evaluate

data_full = subset2evaluate.utils.load_data_wmt(year="wmt23", langs="en-cs")
data_new = subset2evaluate.select_subset.run_select_subset(data_old, method="random")
print(utils.eval_subset_accuracy(data_new, data_full))
```

and compare it to something better:
```python
data_full = subset2evaluate.utils.load_data_wmt(year="wmt23", langs="en-cs")
data_new = subset2evaluate.select_subset.run_select_subset(data_old, method="var", metric="MetricX-23")
print(utils.eval_subset_accuracy(data_new, data_full))
```

# Example for Custom Dataset

TODO process this

```
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