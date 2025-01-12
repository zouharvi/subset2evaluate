# <img src="misc/logo.svg" height="25em"> subset2evaluate

Package to select informative samples to human-evaluate for NLG tasks such as machine translation or summarization.

> TODO abstract


General recommendations based on MT evaluation:

| When to use? | What is it? | How to use? |
|-|-|-|
| Good automated metric available, such as `MetricX-23`. | Variance in metric scores. | `method="var", metric="MetricX-23"` |
| Metric not available but system outputs available. | Diversity of system outputs. | `method="diversity"` |
| System outputs not available, only sources. | Estimated diversity in system outputs. | `method="precomet_diversity"` |

## Example for Machine Translation

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

Benchmarks for some of the methods on WMT23:

<!-- TODO: -->

## Example for Custom Dataset

We encourage testing 
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

## Command-line Interface

We recommend using the Python interface but the package can also be used from the command line:

```
subset2evaluate wmt23/en-cs --method var --args "{'metric': 'MetricX-23'}" > wmt23_encs_sorted.jsonl
subset2evaluate-eval wmt23/en-cs wmt23_encs_sorted.jsonl 
> Clusters: 2.30
> Accuracy: 86.7%
```

## Contact & Contributions

We are look forward to contributions, especially (1) using subset2evaluate for other tasks, (2) adding new methods, (3) finding bugs and increasing package usability.
Please file a GitHub issue or [send us an email](mailto:vilem.zouhar@gmail.com).