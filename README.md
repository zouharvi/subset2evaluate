# subset2evaluate

[![Paper](https://img.shields.io/badge/📜%20paper-481.svg)](https://arxiv.org/abs/2501.18251)
&nbsp;
[![PyPi version](https://badgen.net/pypi/v/subset2evaluate/)](https://pypi.org/project/subset2evaluate)
&nbsp;
[![PyPI download/month](https://img.shields.io/pypi/dm/subset2evaluate.svg)](https://pypi.python.org/pypi/subset2evaluate/)
&nbsp;
[![PyPi license](https://badgen.net/pypi/license/subset2evaluate/)](https://pypi.org/project/subset2evaluate/)
&nbsp;
[![Testing](https://github.com/zouharvi/subset2evaluate/actions/workflows/test.yml/badge.svg)](https://github.com/zouharvi/subset2evaluate/actions/workflows/test.yml)
&nbsp;
[![Hugging Face Collection](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Collection-FCD21D)](https://huggingface.co/collections/zouharvi/precomet-67b5e9ce9782a5e2fd289268)
<hr>

Package to select informative samples to human-evaluate for NLG tasks such as machine translation or summarization.
It is based on work of Vilém Zouhar, Peng Cui, and Mrinmaya Sachan from ETH Zürich.

> **Title:** [How to Select Datapoints for Efficient Human Evaluation of NLG Models?](https://arxiv.org/abs/2501.18251)
> 
> **Abstract:** 
> Human evaluation is the gold-standard for evaluating text generation models.
> It is also expensive, and to fit budgetary constraints, a random subset of the test data is often chosen in practice.
> The randomly selected data may not accurately represent test performance, making this approach economically inefficient for model comparison.
> Thus, in this work, we develop a suite of selectors to get the most informative datapoints for human evaluation while taking the evaluation costs into account.
> We show that selectors based on variance in automated metric scores, diversity in model outputs, or Item Response Theory outperform random selection. 
> We further develop an approach to distill these selectors to the scenario where the model outputs are not yet available.
> In particular, we introduce source-based estimators, which predict item usefulness for human evaluation just based on the source texts.
> We demonstrate the efficacy of our selectors in two common NLG tasks, machine translation and summarization, and show that up to only ~50% of the test data is needed to produce the same evaluation result as the entire data.
> Our implementations are published in the [subset2evaluate](https://pypi.org/project/subset2evaluate/) package.

<img src="https://raw.githubusercontent.com/zouharvi/subset2evaluate/refs/heads/main/misc/highlevel_subset_selection.svg" width="1000em">

## Usage

In short, you put list of items in the package and the package sorts the list in descending order (first is better) based on how suitable each item is for evaluation, such as with human annotations.
In addition to the sorting, the package also returns the item utility stored in the `subset2evaluate_utility` field of each item.
General recommendations based on MT evaluation:

| When to use? | What is it? | How to use? |
|-|-|-|
| Good automated metric available, such as `MetricX-23`. | Variance in metric scores. | `method="metric_var", metric="MetricX-23"` |
| Metric not available but model outputs available. | Diversity of model outputs. | `method="diversity", method="BLEU"` |
| Model outputs not available, only sources. | Estimated diversity in model outputs. | `method="precomet_diversity"` |

The package supports multiple methods.
We show benchmark of the methods on machine translation evaluation.
For the metric-based methods, the results use MetricX-23 but others can be easily used if supplied in the input data.

| Method | Function signature | Requirements | Correlation | Clusters |
|-|-|-|-|-|
| Random | `method="random"` | | 92.5% | 2.25
| **Output-based selection** |
| Metric variance | `method="metric_var", metric="MetricX-23"` | Metric scores | 93.8% | 3.22
| Metric average | `method="metric_avg", metric="MetricX-23"` | Metric scores | 92.9% | 3.16 |
| Metric consistency | `method="metric_cons", metric="MetricX-23"` | Metric scores | 94.2% | 3.24 |
| Diversity BLEU | `method="diversity", metric="BLEU"` | Outputs | 94.0% | 2.99 |
| Diversity unigram | `method="diversity", metric="unigram"` | Outputs | 92.5% | 2.62 |
| Diversity [LM](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) | `method="diversity", metric="lm"` | Outputs | 93.9% | 2.81 |
| DiffDisc | `method="pyirt_diffdisc", metric="MetricX-23"` | Metric scores | 93.7% | 2.83 |
| [DiffUse](https://aclanthology.org/2024.acl-long.456.pdf) | `method="diffuse"` | Outputs | 93.8% | 2.18 |
| **Source-based selection** |
| Var<sup>SRC</sup> [model](https://huggingface.co/zouharvi/PreCOMET-var) | `method="precomet_var"` | Sources | 92.7% | 2.62 |
| Avg<sup>SRC</sup> [model](https://huggingface.co/zouharvi/PreCOMET-avg) | `method="precomet_avg"` | Sources | 92.2% | 2.68 |
| Diversity<sup>SRC</sup> [model](https://huggingface.co/zouharvi/PreCOMET-diversity) | `method="precomet_diversity"` | Sources | 94.0% | 2.86 |
| DiffDisc<sup>SRC</sup> [model](https://huggingface.co/zouharvi/PreCOMET-diffdisc_direct) | `method="precomet_diffdisc_direct"` | Sources | 93.4% | 2.98 |
| Consistency<sup>SRC</sup> [model](https://huggingface.co/zouharvi/PreCOMET-cons) | `method="precomet_cons"` | Sources | 93.8% | 2.77 |
| [Sentinel-SRC-DA](https://huggingface.co/sapienzanlp/sentinel-src-da) | `method="sentinel_src"` | Sources | 92.7% | 2.83 |
| [Sentinel-SRC-MQM](https://huggingface.co/sapienzanlp/sentinel-src-mqm) | `method="sentinel_src_mqm"` | Sources | 92.9% | 3.00 |


And benchmark of the methods for summarization.
For metric-based methods we use coverage but others can be easily used if supplied in the input data.

| Method | Function signature | Requirements | Correlation | Clusters |
|-|-|-|-|-|
| Random | `method="random"` | | 93.5% | 2.14 |
| **Output-based selection** |
| Metric variance | `method="metric_var", metric="Coverage"` | Metric scores | 96.8% | 2.50 |
| Metric average | `method="metric_avg", metric="Coverage"` | Metric scores | 95.7% | 2.30 |
| Metric consistency | `method="metric_cons", metric="Coverage"` | Metric scores | 96.4% | 2.00 |
| DiffDisc | `method="pyirt_diffdisc", metric="Coverage"` | Metric scores | 92.8% | 2.02 |
| Diversity BLEU | `method="diversity", metric="BLEU"` | Outputs | 93.6% | 2.60 |
| Diversity unigram | `method="diversity", metric="unigram"` | Outputs | 91.4% | 2.70 |
| Diversity [LM](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) | `method="diversity", metric="lm"` | Outputs | 97.0% | 2.90 |

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

# take only top 100 items to "human-evaluate"
data_new = subset2evaluate.select_subset.basic(data_full, method="random")
subset2evaluate.evaluate.eval_subset_clusters(data_new[:100])
> 1

# compare it to something better:
data_new = subset2evaluate.select_subset.basic(data_full, method="metric_var", metric="MetricX-23")
subset2evaluate.evaluate.eval_subset_clusters(data_new[:100])
> 3
```

## Example for Summarization

```python
import subset2evaluate

data_full = subset2evaluate.utils.load_data("summeval")
len(data_full)
> 100

# take only top 25 items to "human-evaluate"
data_new = subset2evaluate.select_subset.basic(data_full, method="random")
subset2evaluate.evaluate.eval_subset_clusters(data_new[:25], metric="human_relevance")
> 2

data_new = subset2evaluate.select_subset.basic(data_full, method="diversity", metric="BLEU")
subset2evaluate.evaluate.eval_subset_clusters(data_new[:25], metric="human_relevance")
> 3
```

## Example for Custom Dataset

The intended usage is for your own custom datasets where you wish to choose which to evaluate.
The input to subset2evaluate needs to be a list of items.
What each item needs to contain depends on the method.
For example, `diversity` requires `tgt` on each item such that the output diversity can be computed.
As another example `var` requires `scores/metric` on each item such that the metric variance can be computed.
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
subset2evaluate wmt23/en-de --method metric_var --args "{'metric': 'MetricX-23'}" > wmt23_ende_sorted.jsonl
subset2evaluate-eval wmt23/en-de wmt23_ende_sorted.jsonl 
> Correlation: 87.1%
> Clusters: 2.70
```

## Advanced Usage

The package also supports cost-aware subset selection, which is useful for the cases where we know the estimated annotation costs of items.
For example, annotating a five paragraph-long summarization output likely takes 3-6 times more than a single paragraph output.
For cost-aware selection, the package requires two things:
1. the data has already been ran through `select_subset.basic` method (such that each item now has `subset2evaluate_utility` property), ans
2. each item has a `cost` value
The WMT data already have the cost values for each item (estimated annotation time):
```python
import subset2evaluate
data_full = subset2evaluate.utils.load_data("wmt23/en-zh")
data_full[0]["cost"]
> 0.2973610038416405

# run basic selection
data_new = subset2evaluate.select_subset.basic(data_full, method="metric_var", metric="MetricX-23")

# only the first 23 items fit our budget of 50
sum([line["cost"] for line in data_new[:23]])
> 49.18571270950981

subset2evaluate.evaluate.eval_subset_correlation(data_new[:23], data_full)
> 0.8714285714285712

# let's run cost-aware selection
data_costaware = subset2evaluate.select_subset.costaware(data_new, budget=50)

# indeed the whole output fits our budget
sum([line["cost"] for line in data_costaware])
> 49.98968875693353

subset2evaluate.evaluate.eval_subset_correlation(data_costaware, data_full)
> 0.9107142857142855
```

## Contact & Contributions

We are look forward to contributions, especially (1) using subset2evaluate for other tasks, (2) adding new methods, (3) finding bugs and increasing package usability.
Please file a GitHub issue or [send us an email](mailto:vilem.zouhar@gmail.com).
Some methods from other works have already found their way into subset2evaluate, such as [DiffUse](https://aclanthology.org/2024.acl-long.456.pdf) or [Sentinel Metrics](https://huggingface.co/sapienzanlp/sentinel-src-mqm).

The repository is structured as follows:
- `subset2evaluate/` contains the primary package and all methods 
- `experiments/` contains scripts to run experiments in the paper

Cite as:
```
@misc{zouhar2025selectdatapointsefficienthuman,
    title={How to Select Datapoints for Efficient Human Evaluation of NLG Models?}, 
    author={Vilém Zouhar and Peng Cui and Mrinmaya Sachan},
    year={2025},
    eprint={2501.18251},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    url={https://arxiv.org/abs/2501.18251}, 
}
```
