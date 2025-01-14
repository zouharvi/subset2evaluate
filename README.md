# <img src="https://raw.githubusercontent.com/zouharvi/subset2evaluate/refs/heads/main/misc/logo.svg" height="25em"> subset2evaluate &nbsp;&nbsp;&nbsp; [![PyPI Version](https://img.shields.io/pypi/v/subset2evaluate)](https://pypi.org/project/subset2evaluate/) [![test subset2evaluate](https://github.com/zouharvi/subset2evaluate/actions/workflows/test.yml/badge.svg)](https://github.com/zouharvi/subset2evaluate/actions/workflows/test.yml)

Package to select informative samples to human-evaluate for NLG tasks such as machine translation or summarization.
It is based on a [paper](https://vilda.net/papers/subset2evaluate.pdf) by Vilém Zouhar, Peng Cui, and Mrinmaya Sachan from ETH Zürich.

> [Selecting Examples to Efficiently Human-Evaluate Models](https://vilda.net/papers/subset2evaluate.pdf): Researchers create test sets to evaluate models with human annotations, which are the gold standard as opposed to automated metrics.
> Natural language generation, a rapidly evolving field, has a recurring need for new test sets.
> Oftentimes, to fit the budgetary constraints, only a random subset of the test set is chosen for evaluation.
> This is grossly inefficient and in this work we provide methods to strategically select the most informative samples to be evaluated.
> We describe variance- and diversity-based methods for when we know the system outputs and their evaluation with automated metrics beforehand.
> These methods consistently outperform random subset selection, the most common approach.
> We introduce PreCOMET to make our methods applicable to blind test sets, where the systems are unknown in advance.
> The model is trained to predict item utility for human evaluation just based on the source alone.
> We show on two natural language generation tasks, machine translation and summarization, that these methods make human evaluation more efficient and reduce costs without burdening annotators.

<img src="https://raw.githubusercontent.com/zouharvi/subset2evaluate/refs/heads/main/misc/highlevel_subset_selection.svg" width="1000em">

## Usage

In short, you put list of items in the package and the package sorts the list in descending order (first is better) based on how suitable each item is for evaluation, such as with human annotations.
In addition to the sorting, the package also returns the item utility stored in the `subset2evalute_utility` field of each item.
General recommendations based on MT evaluation:

| When to use? | What is it? | How to use? |
|-|-|-|
| Good automated metric available, such as `MetricX-23`. | Variance in metric scores. | `method="metric_var", metric="MetricX-23"` |
| Metric not available but system outputs available. | Diversity of system outputs. | `method="diversity_bleu"` |
| System outputs not available, only sources. | Estimated diversity in system outputs. | `method="precomet_diversity"` |

The package supports multiple methods.
We show benchmark of the methods on machine translation evaluation:

| Method | Requirements | Accuracy | Cluster count |
|-|-|-|-|
| Random | | 91.0% | 2.25 |
| **Output-based selection** |
| MetricX-23 var | MetricX-23 scores | 92.0% | 3.22 |
| MetricX-23 avg | MetricX-23 scores | 91.8% | 3.16 |
| Diversity BLEU | Outputs | 92.1% | 2.99 |
| Diversity unigram | Outputs | 91.1% | 2.62 |
| IRT diff.×disc. | MetricX-23 scores | 91.2% | 3.14 |
| **Source-based selection** |
| PreCOMET var [model](https://huggingface.co/zouharvi/PreCOMET-var) | Sources | 91.2% | 2.58 |
| PreCOMET avg [model](https://huggingface.co/zouharvi/PreCOMET-avg) | Sources | 91.1% | 2.68 |
| PreCOMET diversity [[model](https://huggingface.co/zouharvi/PreCOMET-diversity)] | Sources | 92.1% | 2.86 |
| PreCOMET diff.×disc. [[model1](https://huggingface.co/zouharvi/PreCOMET-diff), [model2](https://huggingface.co/zouharvi/PreCOMET-disc)] | Sources | 93.1% | 3.22 |


And benchmark of the methods for summarization:

| Method | Requirements | Accuracy | Cluster count |
|-|-|-|-|
| Random | | 90.5% | 2.00 |
| **Output-based selection** |
| Coverage var | Coverage scores | 92.2% | 2.30 |
| Coverage avg | Coverage scores | 91.8% | 2.20 |
| IRT diff.×disc. | Coverage scores | 92.6% | 2.44 |
| Diversity BLEU | Outputs | 89.3% | 2.90 |
| Diversity unigram | Outputs | 87.2% | 2.80 |

## Example for Machine Translation

Install the package and download WMT data:
```bash
pip3 install subset2evaluate
# optionally these two packages for IRT and PreCOMET based selections
pip3 install git+https://github.com/zouharvi/PreCOMET.git git+https://github.com/zouharvi/py-irt.git
```

Then in Python we compute the baseline:
```python
import subset2evaluate

data_full = subset2evaluate.utils.load_data("wmt23/en-cs")
len(data_full)
> 1098

# take only top 100 segments to "human-evaluate"
data_new = subset2evaluate.select_subset.run_select_subset(data_full, method="random")
subset2evaluate.evalute.eval_subset_clusters(data_new[:100])
> 1

# compare it to something better:
data_new = subset2evaluate.select_subset.run_select_subset(data_full, method="metric_var" metric="MetricX-23")
subset2evaluate.evaluate.eval_subset_clusters(data_new[:100])
> 3
```

## Example for Summarization

```python
import subset2evaluate

data_full = subset2evaluate.utils.load_data("summeval")
len(data_full)
> 100

# take only top 25 segments to "human-evaluate"
data_new = subset2evaluate.select_subset.run_select_subset(data_full, method="random")
subset2evaluate.evaluate.eval_subset_clusters(data_new[:25], metric="human_relevance")
> 2

data_new = subset2evaluate.select_subset.run_select_subset(data_full, method="diversity_bleu")
subset2evaluate.evaluate.eval_subset_clusters(data_new[:25], metric="human_relevance")
> 3
```

## Example for Custom Dataset

The intended usage is for your own custom datasets where you wish to choose which to evaluate.
The input to subset2evaluate needs to be a list of items.
What each item needs to contain depends on the method.
For example, `diversity` requires `tgt` on each item such that the output diversity can be computed.
As another texample `var` requires `scores/metric` on each item such that the metric variance can be computed.
The item can contain any additional extra fields even if they're not explicitly used.
As an example, look at the existing loaders:

```python
import subset2evaluate
import json
data = subset2evaluate.utils.load_data("wmt23/en-de")

len(data)
> 549

json.dumps(data[0], indent=2)
> {
>   "i": 0,
>   "src": "Police arrest 15 after violent protest outside UK refugee hotel",
>   "ref": "Polizei verhaftet 15 Menschen nach gewalttätigen Protesten vor einer Flüchtlingsunterkunft in Großbritannien",
>   "tgt": {
>     "Lan-BridgeMT": "Polizei verhaftet 15 nach gewalttätigem Protest vor britischem Flüchtlingshotel",
>     "NLLB_MBR_BLEU": "Polizei verhaftet 15 nach gewaltsamen Protesten vor einem britischen Flüchtlingshotel",
>     "ZengHuiMT": "Die Polizei verhaftet 15 Personen nach gewalttätigem Protest vor britischem Flüchtlingshotel.",
>     "ONLINE-A": "Polizei nimmt 15 nach gewalttätigen Protesten vor britischem Flüchtlingshotel fest",
>     "ONLINE-W": "Polizei nimmt 15 Personen nach gewaltsamen Protesten vor einem britischen Flüchtlingshotel fest",
>     "ONLINE-B": "Polizei verhaftet 15 Personen nach gewalttätigem Protest vor britischem Flüchtlingshotel",
>     "NLLB_Greedy": "Polizei verhaftet 15 nach gewalttätigen Protesten vor einem Flüchtlingshotel in Großbritannien",
>     "ONLINE-M": "Polizei verhaftet 15 nach gewalttätigem Protest vor britischem Flüchtlingshotel",
>     "AIRC": "﻿Polizeiverhaftung 15 nach gewaltsamen Protesten außerhalb des britischen Flüchtlingshotels",
>     "ONLINE-Y": "Die Polizei verhaftet 15 Personen nach gewaltsamen Protesten vor einem britischen Flüchtlingshotel",
>     "GPT4-5shot": "Die Polizei nimmt 15 Personen nach gewalttätigen Protesten vor einem britischen Flüchtlingshotel fest.",
>     "ONLINE-G": "Polizei verhaftet 15 nach gewalttätigem Protest vor britischem Flüchtlingshotel"
>   },
>   "time": 0.2119810263850096,
>   "domain": "news",
>   "doc": "aj-english.33941",
>   "scores": {
>     "Lan-BridgeMT": {
>       "human": 0.9175257731958762,
>       "XCOMET-XL": 0.9867596612701105,
>       "f200spBLEU": 0.2759278681802151,
>       ...
>     },
>     "GPT4-5shot": {
>       "human": 0.9948453608247423,
>       "XCOMET-XL": 0.988012809964431,
>       "f200spBLEU": 0.3275118410766353,
>       ...
>     },
>     "ONLINE-G": {
>       "human": 0.8762886597938144,
>       "XCOMET-XL": 0.9867596612701105,
>       "f200spBLEU": 0.2759278681802151,
>       ...
>     }
>   }
> }

```

## Command-line Interface

We recommend using the Python interface but the package can also be used from the command line:

```
subset2evaluate wmt23/en-cs --method metric_var --args "{'metric': 'MetricX-23'}" > wmt23_encs_sorted.jsonl
subset2evaluate-eval wmt23/en-cs wmt23_encs_sorted.jsonl 
> Clusters: 2.30
> Accuracy: 86.7%
```

## Contact & Contributions

We are look forward to contributions, especially (1) using subset2evaluate for other tasks, (2) adding new methods, (3) finding bugs and increasing package usability.
Please file a GitHub issue or [send us an email](mailto:vilem.zouhar@gmail.com).

The repository is structured as follows:
- `subset2evaluate/` contains the primary package and all methods 
- `experiments/` contains scripts to run experiments in the paper
