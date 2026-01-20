from typing import Any, Callable, Dict, List, Optional, Union, Literal
import numpy as np
from subset2evaluate.reference_info import year2std_refs

PROPS = np.linspace(0.05, 0.5, 10)


def _data_minmax_normalize(data):
    """
    In-place min-max normalization of all scores
    """
    import collections

    # if we are binarizing, none of this matters
    data_flat = collections.defaultdict(list)
    for line in data:
        for met_all in line["scores"].values():
            for met_k, met_v in met_all.items():
                if met_v is None:
                    continue
                data_flat[met_k].append(met_v)

    # normalize
    data_flat = {k: (min(v), max(v)) for k, v in data_flat.items()}

    for line in data:
        for model, met_all in line["scores"].items():
            for met_k, met_v in met_all.items():
                if met_v is None:
                    continue
                if data_flat[met_k][1] - data_flat[met_k][0] == 0:
                    line["scores"][model][met_k] = 0
                else:
                    # (x-min)/(max-min) normalize
                    line["scores"][model][met_k] = (met_v - data_flat[met_k][0]) / (
                        data_flat[met_k][1] - data_flat[met_k][0]
                    )


def confidence_interval(data, confidence=0.95):
    import scipy.stats

    return scipy.stats.t.interval(
        confidence=confidence, df=len(data) - 1, loc=np.mean(data), scale=np.std(data)
    )


def _data_median_binarize(data):
    """
    In-place median binarization of all scores
    """
    import collections

    data_flat = collections.defaultdict(list)
    for line in data:
        for met_all in line["scores"].values():
            for met_k, met_v in met_all.items():
                data_flat[met_k].append(met_v)

    # normalize
    data_flat = {k: np.median(v) for k, v in data_flat.items()}

    for line in data:
        for model, met_all in line["scores"].items():
            for met_k, met_v in met_all.items():
                line["scores"][model][met_k] = 1 * (met_v >= data_flat[met_k])


def mt_esa_eval_cost(src_text: str, langs: str) -> float:
    """
    Returns the cost of the MT ESA evaluation.
    The cost is calculated as 0.15 * word_count + 33.7.
    It's just a very rough estimate, and values should be normalized for cost-aware selection.
    """
    # TODO: Add ref to the paper/dataset

    # for CJK languages, we need to count characters
    if any(cjk in langs for cjk in ("ja", "zh", "ko", "th")):
        word_count = len(src_text.strip())
    else:
        word_count = len(src_text.strip().split())

    # the coefficients don't matter because it should be normalized before use in cost-aware selection
    return 0.15 * word_count + 33.7


def mqm_score(severities, weights: Dict[str, int] = None) -> int:
    """
    Calculate the MQM score based on the errors.

    Args:
        errors (list): List of error dictionaries.

    Returns:
        int: The calculated MQM score.
    """
    if weights is None:
        weights = {
            "neutral": 0,
            "minor": -1,
            "major": -5,
            "critical": -25,
        }
    score = 0
    for severity in severities:
        score += weights.get(severity, 0)
    return score


def get_errors_severities(
    errors: List[Dict[str, Any]], sev_attr_name="severity"
) -> List[str]:
    """
    Extract the severity levels from the errors.

    Args:
        errors (list): List of error dictionaries.
        sev_attr_name (str): The attribute name for severity in the error dictionaries. Default is "severity".

    Returns:
        list: List of severity levels.
    """
    if not errors:
        return []
    return [err.get(sev_attr_name, "").lower() for err in errors]


def ensure_wmt_exists():
    import requests
    import os
    import tarfile

    if not os.path.exists("data/mt-metrics-eval-v2/"):
        print("Downloading WMT data because data/mt-metrics-eval-v2/ does not exist..")
        os.makedirs("data/", exist_ok=True)
        r = requests.get(
            "https://storage.googleapis.com/mt-metrics-eval/mt-metrics-eval-v2.tgz"
        )
        with open("data/mt-metrics-eval-v2.tgz", "wb") as f:
            f.write(r.content)
        with tarfile.open("data/mt-metrics-eval-v2.tgz", "r:gz") as f:
            f.extractall("data/")
        os.remove("data/mt-metrics-eval-v2.tgz")


def load_data_hf(
    dataset: Any,
    loader: Callable = None,
    loader_kwargs: Optional[dict] = None,
    cache_fname: Optional[str] = None,
    converter: Optional[Callable] = None,
    converter_kwargs: Optional[dict] = None,
):
    """Load a dataset from Hugging Face, with optional caching and conversion.
    This function loads a dataset using the provided loader function
    by default `datasets.load_dataset` from the Hugging Face library.
    It also allows for a custom converter function to be applied to the
    dataset after loading.
    It supports caching the loaded and processed dataset to a pickle file to speed up subsequent loads.
    Args:
        dataset (Any): The name of the Hugging Face Dataset to load or the first positional
            argument of the `loader` function.
            This is typically in the format 'username/dataset_name' or just 'dataset_name'.
        loader (Callable, optional): A callable function to load the dataset.
            If None, by default, it uses `datasets.load_dataset` from the Hugging Face library.
            This function should accept the `dataset` as its first argument.
            If you want to use a different loader, you can pass it here.
            Some alternative loaders could be found in:
            https://huggingface.co/docs/datasets/package_reference/loading_methods
            https://huggingface.co/docs/datasets/loading
        loader_kwargs (Optional[dict], optional): Additional keyword arguments to pass
            directly to the `loader` function. Defaults to None.
        cache (Optional[str], optional): Path to a pickle file where the loaded
            and potentially converted dataset will be cached.
            If a string path is provided, it's used for loading from and saving to the cache.
            Defaults to None, meaning no caching is performed.
        converter (Optional[Callable], optional): A callable function that takes
            the loaded Hugging Face dataset as input and transforms it into a
            format suitable for further processing by 'subset2eval'.
            Defaults to None, meaning no conversion is applied.
        converter_kwargs (Optional[dict], optional): Additional keyword arguments
            to pass to the `converter` function if one is provided. Defaults to None.
    Returns:
        Any: The loaded dataset. If a converter is provided, this will be the
             result of the converter. Otherwise, it's the raw dataset loaded
             from Hugging Face.
    """
    import datasets

    if loader is None:
        loader = datasets.load_dataset

    if cache_fname:
        import contextlib
        import importlib
        import os
        import pickle

        # temporarily change to the root directory, this requires Python 3.11
        with contextlib.chdir(os.path.dirname(os.path.realpath(__file__)) + "/../"):
            os.makedirs("data/cache/", exist_ok=True)
            cache_f = f"data/cache/{cache_fname}.pkl"

            # load cache if exists
            if os.path.exists(cache_f):
                with open(cache_f, "rb") as f:
                    cache = pickle.load(f)
                    # only load data if they come from the same version
                    if (
                        isinstance(cache, dict)
                        and "version" in cache.keys()
                        and cache["version"]
                        == importlib.metadata.version("subset2evaluate")
                    ):
                        return cache["data"]

    raw_data = loader(dataset, **(loader_kwargs or {}))

    if converter:
        data = converter(raw_data, **(converter_kwargs or {}))
    else:
        data = raw_data

    if cache_fname:
        # save cache
        with open(cache_f, "wb") as f:
            pickle.dump(
                {
                    "version": importlib.metadata.version("subset2evaluate"),
                    "data": data,
                },
                f,
            )

    return data


def load_data_qe4pe(
    task: Literal["pretask", "main", "posttask"] = "main",
    normalize: bool = False,
    cache_fname: str = None,
    dataset: str = "gsarti/qe4pe",
    mqm_weights: Dict[str, int] = None,
    skip_has_issue: bool = False,
    skip_has_added_critical_error: bool = False,
    score_attrs: List[str] = None,
    mqm_attrs: List[str] = None,
    normalize_attrs: List[str] = None,
    hf_loader_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load the QE4PE dataset for the specified task.

    Args:
        task (str): The task to load data for. Options are 'pretask', 'main', or 'posttask'.
        dataset (str): The name of the dataset to load. Default is "gsarti/qe4pe".
        normalize (bool): Whether to normalize the scores (normalize_attrs and mqm_attrs).
        cache_fname (str): The name of the cache file to use for storing the loaded dataset.
        mqm_weights (Dict[str, int]): Weights for MQM error severities.
        skip_has_issue (bool): Whether to skip entries with issues.
        skip_has_added_critical_error (bool): Whether to skip entries with added critical errors.
        score_attrs (List[str]): Attributes to consider for scores. If None, uses all boolean and numeric attributes
        from the dataset from the 11th column (has_issues) and forward, except those in `mqm_scores`, strings and ids.
        mqm_attrs (List[str]): Attributes to consider for MQM scores, which are used to compute the MQM score from the errors.
        If None, uses the default attributes for MQM scores, which are "mt_xcomet_errors", "pe_xcomet_errors", "highlights",
        "qa_mt_mqm_errors" and "qa_pe_mqm_errors" attributes from the dataset.
        normalize_attrs (List[str]): Attributes to normalize. If None, uses all attributes from score_attrs and mqm_attrs.
        hf_loader_kwargs (Optional[Dict[str, Any]]): Additional keyword arguments to pass to the Hugging Face loader.

    Returns:
        Dataset: The loaded dataset in the required format for subset2eval.
        ('qe4pe', 'src_lang-tgt_lang'): List[Dict[str, Any]]
    """
    assert task in ["pretask", "main", "posttask"], (
        "Task must be one of 'pretask', 'main', or 'posttask'."
    )

    if any(attr in (hf_loader_kwargs or {}) for attr in ("name", "split")):
        print(
            f"WARNING: Be aware that passing 'name' or 'split' in hf_loader_kwargs will override QE4PE values ('name'={task}, 'split'=train)."
        )

    import datasets

    def qe4pe_converter(
        hf_dataset: datasets.Dataset,
        normalize: bool = False,
        mqm_weights: Dict[str, int] = None,
        skip_has_issue: bool = False,
        skip_has_added_critical_error: bool = False,
        score_attrs: List[str] = None,
        mqm_attrs: List[str] = None,
        normalize_attrs: List[str] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Convert the Hugging Face dataset to the required format for subset2eval.

        Args:
            hf_dataset (Dataset): The Hugging Face dataset to convert.
            normalize (bool): Whether to normalize the costs and mqm scores.
            mqm_weights (Dict[str, int]): Weights for MQM error severities.
            skip_has_issue (bool): Whether to skip entries with issues.
            skip_has_added_critical_error (bool): Whether to skip entries with added critical errors.
            score_attrs (List[str]): Attributes to consider for scores. If None, uses all boolean and numeric attributes from the dataset from the 11th column (has_issues) and forward, except those in `mqm_scores`, strings and ids.
            mqm_attrs (List[str]): Attributes to consider for MQM scores, which are used to compute the MQM score from the errors. If None, uses the default attributes for MQM scores, which are "mt_xcomet_errors", "pe_xcomet_errors", "highlights", "qa_mt_mqm_errors" and "qa_pe_mqm_errors" attributes from the dataset

        Returns:
            Dict[str, List[Dict[str, Any]]]: The converted dataset.
        """

        import ast
        import collections
        import unicodedata

        attrs_to_ignore = [
            "unit_id",
            "wmt_id",
            "wmt_category",
            "doc_id",
            "segment_in_doc_id",
            "segment_id",
            "translator_pretask_id",
            "translator_main_id",
            "src_lang",
            "tgt_lang",
            "highlight_modality",
            "issue_description",
            "critical_error_description",
            "mt_xcomet_errors",
            "pe_xcomet_errors",
            "src_text",
            "mt_text",
            "mt_text_highlighted",
            "pe_text",
            "mt_pe_word_aligned",
            "mt_pe_char_aligned",
            "highlights",
            "qa_mt_annotator_id",
            "qa_pe_annotator_id",
            "qa_mt_annotated_text",
            "qa_pe_annotated_text",
            "qa_mt_fixed_text",
            "qa_pe_fixed_text",
            "qa_mt_mqm_errors",
            "qa_pe_mqm_errors",
        ]
        mqm_attrs = mqm_attrs or (
            "mt_xcomet_errors",
            "pe_xcomet_errors",
            "highlights",
            "qa_mt_mqm_errors",
            "qa_pe_mqm_errors",
        )
        score_attrs = score_attrs or [
            col_name
            for col_name in hf_dataset.column_names
            if col_name not in [*attrs_to_ignore, *mqm_attrs]
        ]

        grouped = {}
        for item in hf_dataset:
            if skip_has_issue and item.get("has_issue", False):
                continue
            if skip_has_added_critical_error and item.get(
                "has_added_critical_error", False
            ):
                continue
            src = unicodedata.normalize("NFKC", item["src_text"].strip())
            ref = unicodedata.normalize("NFKC", item["mt_text"].strip())
            doc = item["doc_id"]
            seg = item["segment_in_doc_id"]
            domain = item["wmt_category"]
            langs = f"{item['src_lang']}-{item['tgt_lang']}"

            model = item["translator_main_id"]
            tgt = unicodedata.normalize("NFKC", item["pe_text"].strip())

            scores = {
                score: float(item[score]) if item[score] is not None else 0.0
                for score in score_attrs
                if score in item
            }
            # compute score for MQM errors
            for score in mqm_attrs:
                if score in item:
                    scores[score] = (
                        100
                        + mqm_score(
                            get_errors_severities(ast.literal_eval(item[score])),
                            weights=mqm_weights,
                        )
                        if item[score] is not None
                        else 0.0
                    )

            key = (langs, doc, seg)
            if key not in grouped:
                grouped[key] = {
                    "src": src,
                    "ref": ref,
                    "doc": doc,
                    "domain": domain,
                    "langs": langs,
                    "tgt": {},
                    "scores": {},
                }

            grouped[key]["tgt"][model] = tgt
            grouped[key]["scores"][model] = scores

        # convert to final format
        data = collections.defaultdict(list)
        for item in grouped.values():
            cost = mt_esa_eval_cost(item["src"], item["langs"])
            entry = {
                "src": item["src"],
                "ref": item["ref"],
                "tgt": item["tgt"],
                "scores": item["scores"],
                "doc": item["doc"],
                "domain": item["domain"],
                "cost": cost,
            }
            data[("qe4pe", item["langs"])].append(entry)

        # always normalize costs
        if data:
            for data_per_lang in data.values():
                costs = np.array([x["cost"] for x in data_per_lang])
                cost_norm = (costs - costs.mean()) / costs.std() + 1
                cost_norm = (cost_norm - cost_norm.min()) / (1 - cost_norm.min())
                for i, x in enumerate(data_per_lang):
                    x["cost"] = float(cost_norm[i])
                    # also set a 0 based index for each item of each language pair
                    x["i"] = i

            if normalize:
                for data_per_lang in data.values():
                    # Normalize scores to 0–100 if requested
                    for attr in normalize_attrs or [*score_attrs, *mqm_attrs]:
                        all_scores = [
                            score[attr]
                            for item in data_per_lang
                            for score in item["scores"].values()
                            if attr in score
                        ]
                        if not all_scores:
                            continue
                        smin, smax = min(all_scores), max(all_scores)
                        for item in data_per_lang:
                            for model in item["scores"]:
                                raw = item["scores"][model][attr]
                                scaled = (
                                    100 * (raw - smin) / (smax - smin)
                                    if smax > smin
                                    else 0
                                )
                                item["scores"][model][attr] = float(scaled)

        return dict(data)  # Convert defaultdict to dict

    # split is always "train" for QE4PE
    return load_data_hf(
        dataset,
        loader_kwargs={"name": task, "split": "train"} | (hf_loader_kwargs or {}),
        cache_fname=cache_fname,
        converter=qe4pe_converter,
        converter_kwargs={
            "normalize": normalize,
            "mqm_weights": mqm_weights,
            "skip_has_issue": skip_has_issue,
            "skip_has_added_critical_error": skip_has_added_critical_error,
            "score_attrs": score_attrs,
            "mqm_attrs": mqm_attrs,
            "normalize_attrs": normalize_attrs,
        },
    )


def load_data_biomqm(
    split: Literal["dev", "test"] = "dev",
    normalize: bool = False,
):
    import json
    import os
    import collections
    import contextlib

    assert split in {"dev", "test"}, "split must be either 'dev' or 'test'"

    # temporarily change to the root directory, this requires Python 3.11
    with contextlib.chdir(os.path.dirname(os.path.realpath(__file__)) + "/../"):
        # ensure BioMQM exists
        if not os.path.exists(f"data/biomqm/{split}.jsonl"):
            import requests

            print(
                f"Downloading {split} BioMQM data because data/biomqm/{split} does not exist.."
            )
            os.makedirs("data/biomqm", exist_ok=True)

            r = requests.get(
                f"https://huggingface.co/datasets/zouharvi/bio-mqm-dataset/resolve/main/{split}.jsonl?download=true"
            )
            with open(f"data/biomqm/{split}.jsonl", "wb") as f:
                f.write(r.content)

        with open(f"data/biomqm/{split}.jsonl", "r") as f:
            lines = [json.loads(line) for line in f]

    grouped = {}
    for item in lines:
        src = item["src"]
        # for compatibility with WMT loader
        ref = item["ref"][0].strip() if len(item["ref"]) > 0 else None
        doc = item.get("doc_id", "bio-doc")
        domain = "bio"
        langs = f"{item['lang_src']}-{item['lang_tgt']}"

        model = item["system"]
        tgt = item["tgt"].strip()

        # compute score
        score = 100
        for err in item["errors_tgt"]:
            severity = err.get("severity", "").lower()
            if severity == "minor":
                score -= 1
            elif severity == "major":
                score -= 5

        key = (src, ref, doc)
        if key not in grouped:
            grouped[key] = {
                "src": src,
                "ref": ref,
                "doc": doc,
                "domain": domain,
                "langs": langs,
                "tgt": {},
                "scores": {},
            }

        grouped[key]["tgt"][model] = tgt
        grouped[key]["scores"][model] = {"human": float(score)}

    # convert to list
    data = collections.defaultdict(list)
    for i, item in enumerate(grouped.values()):
        entry = {
            "i": i,
            "src": item["src"],
            "ref": item["ref"],
            "tgt": item["tgt"],
            "scores": item["scores"],
            "doc": item["doc"],
            "domain": item["domain"],
            "cost": mt_esa_eval_cost(item["src"], item["langs"]),
        }
        data[("biomqm", item["langs"])].append(entry)

    if normalize:
        for data_per_lang in data.values():
            # Normalize cost
            costs = np.array([x["cost"] for x in data_per_lang])
            cost_norm = (costs - costs.mean()) / costs.std() + 1
            cost_norm = (cost_norm - cost_norm.min()) / (1 - cost_norm.min())
            for i, x in enumerate(data_per_lang):
                x["cost"] = float(cost_norm[i])

            # Normalize scores to 0–100 if requested
            all_scores = [
                score["human"]
                for item in data_per_lang
                for score in item["scores"].values()
            ]
            smin, smax = min(all_scores), max(all_scores)
            for item in data_per_lang:
                for model in item["scores"]:
                    raw = item["scores"][model]["human"]
                    scaled = 100 * (raw - smin) / (smax - smin) if smax > smin else 0
                    item["scores"][model]["human"] = float(scaled)

    return data


def load_data_wmt(  # noqa: C901
    year: str = "wmt23",
    langs: str = "en-cs",
    normalize: bool = False,
    binarize: bool = False,
    file_protocol: Optional[str] = None,
    file_reference: Optional[str] = None,
    zero_bad: bool = False,
    include_ref: bool = False,
    require_human: bool = True,
):
    """
    Load WMT data for a specific year and language pair.

    Args:
        year (str): WMT year (e.g. "wmt23").
        langs (str): Language pair (e.g. "en-cs").
        normalize (bool): Whether to min-max normalize scores of the systems.
        binarize (bool): Whether to binarize scores of the systems (median split).
        file_protocol (Optional[str]): Specific human score file protocol to use (e.g. "mqm", "wmt").
            If None, tries a list of defaults.
        file_reference (Optional[str]): Specific reference identifier to use (e.g. "refA").
            If None, uses the standard reference for that year/lang.
        zero_bad (bool): If True, exclude systems where the human score is "0" (in addition to "None").
        include_ref (bool): Whether to include human references (if found in system outputs) as systems.
        require_human (bool): If True, return empty list if no human scores file found, and filter out lines/models with no human score.
            If False, human scores are optional (defaulting to 0).

    Returns:
        List[Dict[str, Any]]: A list of segments, where each segment is a dict containing
            src, ref, tgt (dict of model outputs), scores (dict of scores), etc.
    """
    import glob
    import collections
    import numpy as np
    import os
    import pickle
    import contextlib
    import importlib.metadata

    # temporarily change to the root directory, this requires Python 3.11
    with contextlib.chdir(os.path.dirname(os.path.realpath(__file__)) + "/../"):
        ensure_wmt_exists()

        os.makedirs("data/cache/", exist_ok=True)
        cache_f = f"data/cache/{year}_{langs}_n{int(normalize)}_b{int(binarize)}_fp{file_protocol}_fr{file_reference}_zb{int(zero_bad)}_ir{int(include_ref)}_rh{int(require_human)}.pkl"

        # load cache if exists
        if os.path.exists(cache_f):
            with open(cache_f, "rb") as f:
                cache = pickle.load(f)
                # only load data if they come from the same version
                if (
                    isinstance(cache, dict)
                    and "version" in cache.keys()
                    and cache["version"]
                    == importlib.metadata.version("subset2evaluate")
                ):
                    return cache["data"]

        lines_src = open(
            f"data/mt-metrics-eval-v2/{year}/sources/{langs}.txt", "r"
        ).readlines()
        lines_doc = open(
            f"data/mt-metrics-eval-v2/{year}/documents/{langs}.docs", "r"
        ).readlines()
        lines_ref = None

        refs_dir = f"data/mt-metrics-eval-v2/{year}/references"
        selected_human_ref = (
            file_reference if file_reference is not None else year2std_refs[year][langs]
        )
        file_reference_path = f"{refs_dir}/{langs}.{selected_human_ref}.txt"

        if not os.path.exists(file_reference_path):
            # did not find reference
            return []

        lines_ref = open(file_reference_path, "r").readlines()

        # collect all human reference names (used for filtering in case include_ref is True)
        pattern = f"{langs}.*.txt"
        ref_files = glob.glob(os.path.join(refs_dir, pattern))
        human_refs = set()
        for filepath in ref_files:
            basename = os.path.basename(filepath)  # es. "en-de.refB.txt"
            parts = basename.split(".")  # -> ["en-de", "refB", "txt"]
            if len(parts) == 3 and parts[0] == langs:
                human_refs.add(parts[1])

        # do not consider canary line
        contain_canary_line = lines_src[0].lower().startswith("canary")
        if contain_canary_line:
            lines_src.pop(0)
            lines_doc.pop(0)
            lines_ref.pop(0)

        line_model = {}
        for f in glob.glob(
            f"data/mt-metrics-eval-v2/{year}/system-outputs/{langs}/*.txt"
        ):
            model = f.split("/")[-1].removesuffix(".txt")
            if model in {"synthetic_ref", "chrf_bestmbr"}:
                continue
            if model in human_refs and not include_ref:
                continue

            line_model[model] = open(f, "r").readlines()
            if contain_canary_line:
                line_model[model].pop(0)

        models = list(line_model.keys())

        lines_score = collections.defaultdict(list)

        if file_protocol is not None:
            f_protocols = [
                f"data/mt-metrics-eval-v2/{year}/human-scores/{langs}.{file_protocol}.seg.score",
            ]
        else:
            f_protocols = [
                f"data/mt-metrics-eval-v2/{year}/human-scores/{langs}.mqm.seg.score",
                f"data/mt-metrics-eval-v2/{year}/human-scores/{langs}.esa-merged.seg.score",
                f"data/mt-metrics-eval-v2/{year}/human-scores/{langs}.esa.seg.score",
                f"data/mt-metrics-eval-v2/{year}/human-scores/{langs}.da-sqm.seg.score",
                f"data/mt-metrics-eval-v2/{year}/human-scores/{langs}.mqm.seg.score",
                f"data/mt-metrics-eval-v2/{year}/human-scores/{langs}.wmt.seg.score",
                f"data/mt-metrics-eval-v2/{year}/human-scores/{langs}.appraise.seg.score",
                f"data/mt-metrics-eval-v2/{year}/human-scores/{langs}.wmt-raw.seg.score",
                f"data/mt-metrics-eval-v2/{year}/human-scores/{langs}.wmt-appraise.seg.score",
            ]
        for fname in [*f_protocols, False]:
            if fname and os.path.exists(fname):
                break

        if not fname:
            if require_human:
                # did not find human scores
                return []
            else:
                # fill with dummy data because we want to load metric scores
                for model in models:
                    lines_score[model] = [{"human": "None"}] * len(lines_src)
        else:
            for line_raw in open(fname, "r").readlines():
                model, score = line_raw.strip().split()
                lines_score[model].append({"human": score})
        if contain_canary_line:
            for model in lines_score:
                lines_score[model].pop(0)

        total_n_srcs = len(lines_src)
        if contain_canary_line:
            total_n_srcs += 1

        for f in glob.glob(
            f"data/mt-metrics-eval-v2/{year}/metric-scores/{langs}/*.seg.score"
        ):
            # among ref-based metrics, load only the scores for the selected human ref
            if not f.endswith(f"-{selected_human_ref}.seg.score") and not f.endswith(
                "-src.seg.score"
            ):
                continue
            # remove suffix for both ref-based and ref-less metrics
            metric = (
                f.split("/")[-1]
                .removesuffix(f"-{selected_human_ref}.seg.score")
                .removesuffix("-src.seg.score")
            )
            for line_i, line_raw in enumerate(open(f, "r").readlines()):
                model, score = line_raw.strip().split("\t")
                # for refA, refB, synthetic_ref, and other "modeltems" not evaluated
                # NOTE: another option is remove the *models*
                if model not in lines_score:
                    if require_human:
                        continue
                    else:
                        lines_score[model] = [{"human": None}] * (len(lines_src))

                model_line_i = line_i % total_n_srcs
                if contain_canary_line:
                    # do not include canary line scores
                    if model_line_i == 0:
                        continue

                    model_line_i -= 1

                lines_score[model][model_line_i][metric] = float(score)

        # filter out lines that have no human score
        lines_score = {k: v for k, v in lines_score.items() if len(v) > 0}
        models = [model for model in models if model in lines_score]

        # putting it all together
        data = []
        line_id_true = 0

        # remove models that have no outputs
        bad_annotation = {"None"}
        if zero_bad:
            bad_annotation |= {"0"}
        models_bad = set()
        if require_human:
            for model, scores in lines_score.items():
                if all([x["human"] in bad_annotation for x in scores]):
                    models_bad.add(model)
        models = [model for model in models if model not in models_bad]

        for line_i, (line_src, line_ref, line_doc) in enumerate(
            zip(lines_src, lines_ref, lines_doc)
        ):
            # filter None on the whole row
            if require_human and any(
                [
                    lines_score[model][line_i]["human"] in bad_annotation
                    for model in models
                ]
            ):
                continue
            # metrics = set(lines_score[models[0]][line_i].keys())
            # # if we're missing some metric, skip the line
            # if any([set(lines_score[model][line_i].keys()) != metrics for model in models]):
            #     continue

            line_domain, line_doc = line_doc.strip().split("\t")
            data.append(
                {
                    "i": line_id_true,
                    "src": line_src.strip(),
                    "ref": line_ref.strip(),
                    "tgt": {
                        model: line_model[model][line_i].strip() for model in models
                    },
                    # just a very rough estimate, the coefficients don't matter because it'll be normalized later anyway
                    "cost": mt_esa_eval_cost(line_src.strip(), langs),
                    "domain": line_domain,
                    "doc": line_doc,
                    "scores": {
                        model: {
                            metric: float(v) if v is not None and v != "None" else None
                            for metric, v in lines_score[model][line_i].items()
                        }
                        for model in models
                    },
                }
            )
            line_id_true += 1

        # normalize times
        if data:
            data_flat = [line["cost"] for line in data]
            cost_avg = np.average(data_flat)
            cost_std = np.std(data_flat)
            for line in data:
                # z-normalize and make mean 1
                line["cost"] = (line["cost"] - cost_avg) / cost_std + 1

            data_flat = [line["cost"] for line in data]
            cost_min = np.min(data_flat)
            for line in data:
                # make sure it's positive
                line["cost"] = (line["cost"] - cost_min) / (1 - cost_min)

        # collect all humeval scores and try to deduce what happened there
        humscores = [
            model_v["human"] for line in data for model_v in line["scores"].values()
        ]
        if all(x <= 0 for x in humscores if x is not None):
            for line in data:
                for model_v in line["scores"].values():
                    if model_v["human"] is not None:
                        model_v["human"] = max(0, min(100, 100 + 4 * model_v["human"]))
        elif all(x >= 0 and x <= 1 for x in humscores if x is not None):
            for line in data:
                for model_v in line["scores"].values():
                    if model_v["human"] is not None:
                        model_v["human"] = max(0, min(100, model_v["human"] * 100))
        else:
            for line in data:
                for model_v in line["scores"].values():
                    if model_v["human"] is not None:
                        model_v["human"] = max(0, min(100, model_v["human"]))

        # this is min-max normalization
        if normalize and not binarize:
            _data_minmax_normalize(data)

        if binarize:
            _data_median_binarize(data)

        # save cache
        with open(cache_f, "wb") as f:
            pickle.dump(
                {
                    "version": importlib.metadata.version("subset2evaluate"),
                    "data": data,
                },
                f,
            )

    return data


def load_data_wmt_test(**kwargs):
    data = {
        args: load_data_wmt(*args, **kwargs)
        for args in [
            ("wmt23", "cs-uk"),
            ("wmt23", "de-en"),
            ("wmt23", "en-cs"),
            ("wmt23", "en-de"),
            ("wmt23", "en-ja"),
            ("wmt23", "en-zh"),
            ("wmt23", "he-en"),
            ("wmt23", "ja-en"),
            ("wmt23", "zh-en"),
        ]
    }
    return data


def load_data_wmt_all(min_items=100, **kwargs):
    data = {
        args: load_data_wmt(*args, **kwargs)
        for args in [
            ("wmt25", "cs-de_DE"),
            ("wmt25", "cs-uk_UA"),
            ("wmt25", "en-ar_EG"),
            ("wmt25", "en-bho_IN"),
            ("wmt25", "en-cs_CZ"),
            ("wmt25", "en-et_EE"),
            ("wmt25", "en-is_IS"),
            ("wmt25", "en-it_IT"),
            ("wmt25", "en-ja_JP"),
            ("wmt25", "en-ko_KR"),
            ("wmt25", "en-mas_KE"),
            ("wmt25", "en-ru_RU"),
            ("wmt25", "en-sr_Cyrl_RS"),
            ("wmt25", "en-uk_UA"),
            ("wmt25", "en-zh_CN"),
            ("wmt25", "ja-zh_CN"),
            ("wmt24", "cs-uk"),
            ("wmt24", "en-cs"),
            ("wmt24", "en-de"),
            ("wmt24", "en-es"),
            ("wmt24", "en-hi"),
            ("wmt24", "en-is"),
            ("wmt24", "en-ja"),
            ("wmt24", "en-ru"),
            ("wmt24", "en-uk"),
            ("wmt24", "en-zh"),
            ("wmt24", "ja-zh"),
            ("wmt23.sent", "en-de"),
            ("wmt23", "cs-uk"),
            ("wmt23", "de-en"),
            ("wmt23", "en-cs"),
            ("wmt23", "en-he"),
            ("wmt23", "en-de"),
            ("wmt23", "en-ja"),
            ("wmt23", "en-ru"),
            ("wmt23", "en-uk"),
            ("wmt23", "en-zh"),
            ("wmt23", "he-en"),
            ("wmt23", "ja-en"),
            ("wmt23", "ru-en"),
            ("wmt23", "uk-en"),
            ("wmt23", "zh-en"),
            ("wmt22", "cs-en"),
            ("wmt22", "cs-uk"),
            ("wmt22", "de-en"),
            ("wmt22", "de-fr"),
            ("wmt22", "en-cs"),
            ("wmt22", "en-de"),
            ("wmt22", "en-hr"),
            ("wmt22", "en-ja"),
            ("wmt22", "en-liv"),
            ("wmt22", "en-ru"),
            ("wmt22", "en-uk"),
            ("wmt22", "en-zh"),
            ("wmt22", "fr-de"),
            ("wmt22", "ja-en"),
            ("wmt22", "liv-en"),
            ("wmt22", "ru-en"),
            ("wmt22", "ru-sah"),
            ("wmt22", "sah-ru"),
            ("wmt22", "uk-cs"),
            ("wmt22", "uk-en"),
            ("wmt22", "zh-en"),
            ("wmt21.tedtalks", "en-de"),
            ("wmt21.tedtalks", "en-ru"),
            ("wmt21.tedtalks", "zh-en"),
            ("wmt21.news", "cs-en"),
            ("wmt21.news", "de-en"),
            ("wmt21.news", "de-fr"),
            ("wmt21.news", "en-cs"),
            ("wmt21.news", "en-de"),
            ("wmt21.news", "en-ha"),
            ("wmt21.news", "en-is"),
            ("wmt21.news", "en-ja"),
            ("wmt21.news", "en-ru"),
            ("wmt21.news", "en-zh"),
            ("wmt21.news", "fr-de"),
            ("wmt21.news", "ha-en"),
            ("wmt21.news", "is-en"),
            ("wmt21.news", "ja-en"),
            ("wmt21.news", "ru-en"),
            ("wmt21.news", "zh-en"),
            ("wmt21.flores", "bn-hi"),
            ("wmt21.flores", "hi-bn"),
            ("wmt21.flores", "xh-zu"),
            ("wmt21.flores", "zu-xh"),
            ("wmt20", "km-en"),
            ("wmt20", "en-zh"),
            ("wmt20", "ps-en"),
            ("wmt20", "zh-en"),
            ("wmt20", "ru-en"),
            ("wmt20", "iu-en"),
            ("wmt20", "ta-en"),
            ("wmt20", "en-ta"),
            ("wmt20", "en-cs"),
            ("wmt20", "de-en"),
            ("wmt20", "en-de"),
            ("wmt20", "en-ja"),
            ("wmt20", "cs-en"),
            ("wmt20", "en-pl"),
            ("wmt20", "pl-en"),
            ("wmt20", "en-ru"),
            ("wmt20", "en-iu"),
            ("wmt20", "ja-en"),
            ("wmt19", "en-zh"),
            ("wmt19", "en-lt"),
            ("wmt19", "en-gu"),
            ("wmt19", "ru-en"),
            ("wmt19", "kk-en"),
            ("wmt19", "en-fi"),
            ("wmt19", "zh-en"),
            ("wmt19", "fi-en"),
            ("wmt19", "en-cs"),
            ("wmt19", "de-en"),
            ("wmt19", "en-ru"),
            ("wmt19", "gu-en"),
            ("wmt19", "en-kk"),
            ("wmt19", "lt-en"),
            ("wmt19", "de-fr"),
            ("wmt19", "fr-de"),
            ("wmt19", "en-de"),
            ("wmt19", "de-cs"),
        ]
    }
    # filter out empty datasets
    # some years/langs have issues with human annotations coverage
    return {k: v for k, v in data.items() if len(v) > min_items}


def load_data_summeval(  # noqa: C901
    normalize: str = True,
    load_extra: str = False,
):
    from datasets import load_dataset
    from functools import reduce
    import collections
    import contextlib
    import os
    import pickle

    # temporarily change to the root directory, this requires Python 3.11
    with contextlib.chdir(os.path.dirname(os.path.realpath(__file__)) + "/../"):
        os.makedirs("data/cache/", exist_ok=True)
        cache_f = f"data/cache/summeval_n{int(normalize)}_l{int(load_extra)}.pkl"

        # load cache if exists
        if os.path.exists(cache_f):
            with open(cache_f, "rb") as f:
                return pickle.load(f)

    data_raw = load_dataset("KnutJaegersberg/summeval_pairs")["train"]

    data_by_id = collections.defaultdict(list)
    for line in data_raw:
        data_by_id[line["id"]].append(line)

    def avg_human_annotations(
        expert_annotations: List[Dict[str, float]],
    ) -> Dict[str, float]:
        scores = collections.defaultdict(list)
        for line in expert_annotations:
            for k, v in line.items():
                scores[k].append(v)
        scores = {"human_" + k: sum(v) / len(v) for k, v in scores.items()}
        scores_values = list(scores.values())

        # multiply all human
        scores["human_sum"] = reduce(lambda x, y: x + y, scores_values)
        scores["human_mul"] = reduce(lambda x, y: x * y, scores_values)
        return scores

    data = []
    for i, v in data_by_id.items():
        data.append(
            {
                "i": i,
                "src": v[0]["text"],
                "ref": None,
                "tgt": {line["model_id"]: line["decoded"] for line in v},
                "scores": {
                    # rouge is nested for some reason
                    line["model_id"]: (
                        line["metric_scores_1"]
                        | line["metric_scores_1"]["rouge"]
                        | avg_human_annotations(line["expert_annotations"])
                    )
                    for line in v
                },
            }
        )

    # remove rouge from scores and fix supert
    data = [
        {
            **line,
            "scores": {
                model: {
                    metric: score if metric != "supert" else score[0]
                    for metric, score in metrics.items()
                    if metric != "rouge"
                }
                for model, metrics in line["scores"].items()
            },
        }
        for line in data
    ]

    if load_extra:
        # temporarily change to the root directory, this requires Python 3.11
        with contextlib.chdir(os.path.dirname(os.path.realpath(__file__)) + "/../"):
            # TODO: in the future these files need to be stored somewhere statically
            data_metrics = load_data(
                "../subset2evaluate-tmp/data_other/summeval_gpt.jsonl"
            )

        data_metrics_i = {x["i"]: x for x in data_metrics}
        assert all(x["i"] in data_metrics_i for x in data)
        for x in data:
            x["scores"] = {
                sys: data_metrics_i[x["i"]]["scores"][sys] | v
                for sys, v in x["scores"].items()
                if sys in data_metrics_i[x["i"]]["scores"]
            }
            x["scores"] = {
                sys: v
                | {
                    "gpt_sum": v["gpt_relevance"]
                    + v["gpt_coherence"]
                    + v["gpt_consistency"]
                    + v["gpt_fluency"],
                    "gpt_mul": v["gpt_relevance"]
                    * v["gpt_coherence"]
                    * v["gpt_consistency"]
                    * v["gpt_fluency"],
                }
                for sys, v in x["scores"].items()
            }

        # temporarily change to the root directory, this requires Python 3.11
        with contextlib.chdir(os.path.dirname(os.path.realpath(__file__)) + "/../"):
            # TODO: in the future these files need to be stored somewhere statically
            data_metrics = load_data(
                "../subset2evaluate-tmp/data_other/summeval_unieval.jsonl"
            )

        data_metrics_i = {x["i"]: x for x in data_metrics}
        assert all(x["i"] in data_metrics_i for x in data)
        for x in data:
            x["scores"] = {
                sys: data_metrics_i[x["i"]]["scores"][sys] | v
                for sys, v in x["scores"].items()
                if sys in data_metrics_i[x["i"]]["scores"]
            }
            x["scores"] = {
                sys: v
                | {
                    "unieval_sum": (
                        v["unieval_relevance"]
                        + v["unieval_coherence"]
                        + v["unieval_consistency"]
                        + v["unieval_fluency"]
                    ),
                    "unieval_mul": (
                        v["unieval_relevance"]
                        * v["unieval_coherence"]
                        * v["unieval_consistency"]
                        * v["unieval_fluency"]
                    ),
                }
                for sys, v in x["scores"].items()
            }

    if normalize:
        _data_minmax_normalize(data)

    # temporarily change to the root directory, this requires Python 3.11
    with contextlib.chdir(os.path.dirname(os.path.realpath(__file__)) + "/../"):
        # save cache
        with open(cache_f, "wb") as f:
            pickle.dump(data, f)

    return data


def load_data_rose():
    import datasets

    data_raw = datasets.load_dataset("Salesforce/rose", "cnndm_protocol")
    data_raw = list(data_raw.values())[0]
    data = []
    for line in data_raw:
        data.append(
            {
                "i": line["example_id"],
                "src": line["source"],
                "ref": line["reference"],
                "tgt": line["model_outputs"],
                # NOTE: no metrics!
                "scores": line["annotations"],
            }
        )


def pred_irt(model_theta, item):
    import numpy as np

    if "feas" in item:
        # NOTE: true for 4PL, not for 3PL
        # return  item["feas"] / (1 + np.exp(-item["disc"] * (model_theta - item["diff"])))
        return item["feas"] + (1 - item["feas"]) / (
            1 + np.exp(-item["disc"] * (model_theta - item["diff"]))
        )
    if "disc" in item:
        return 1 / (1 + np.exp(-item["disc"] * (model_theta - item["diff"])))
    if "diff" in item:
        return 1 / (1 + np.exp(model_theta - item["diff"]))
    raise Exception("Uknown item", item)


def sanitize_data(data: List[Dict], top_systems=5):
    """
    Makes sure that all items contain the same systems.
    """
    import collections

    system_counter = collections.Counter()
    for line in data:
        for system in line["scores"].keys():
            system_counter[system] += 1

    systems = {x[0] for x in system_counter.most_common(top_systems)}

    # filter items that don't have these systems
    data = [
        line
        for line in data
        if (
            all(system in line["scores"] for system in systems)
            and all(system in line["tgt"] for system in systems)
        )
    ]

    # filter systems that are not everywhere
    systems = set(data[0]["scores"].keys())
    for line in data:
        systems = systems.intersection(set(line["scores"].keys()))

    # filter other systems
    data = [
        {
            **line,
            "scores": {
                system: metrics
                for system, metrics in line["scores"].items()
                if system in systems
            },
            "tgt": {
                system: tgt for system, tgt in line["tgt"].items() if system in systems
            },
        }
        for line in data
    ]
    return data


def load_data(data: Union[List, str], **kwargs):
    import os
    import json

    if type(data) is list:
        pass
    elif os.path.exists(data):
        return [json.loads(x) for x in open(data, "r")]
    elif data.startswith("wmt"):
        data_year, data_lang = data.split("/")
        if data_year == "wmt" and data_lang == "all":
            data = load_data_wmt_all(**kwargs)
        else:
            data = load_data_wmt(year=data_year, langs=data_lang, **kwargs)
    elif data == "summeval":
        return load_data_summeval(**kwargs)
    elif data == "biomqm":
        return load_data_biomqm(**kwargs)
    elif data == "qe4pe":
        return load_data_qe4pe(**kwargs)
    else:
        raise Exception("Could not parse data")

    return data


def load_data_iwslt(compute_automated_metrics=False):
    import json
    import os
    import collections

    data_all = {}

    for langs in ["en-ar", "en-de", "en-zh"]:
        with open(
            os.path.dirname(os.path.realpath(__file__))
            + f"/../data/iwslt/iwslt2025.{langs.replace('-', '')}.shuffled.jsonl",
            "r",
        ) as f:
            lines = [json.loads(line) for line in f]
            lines_meta = {line["id"]: line["meta"] for line in lines}
            lines_prev = {line["id"]: line["previous_context"] for line in lines}

        with open(
            os.path.dirname(os.path.realpath(__file__))
            + f"/../data/iwslt/Final-iwslt2025.{langs.replace('-', '')}.shuffled.jsonl",
            "r",
        ) as f:
            lines = [json.loads(line) for line in f]
            data = collections.defaultdict(list)
            import collections

            tmp_prevcontext = collections.Counter()

            for line in lines:
                line["meta"] = lines_meta[int(line["id"])]
                tmp = lines_prev[int(line["id"])]
                if tmp is None:
                    tmp_prevcontext["None"] += 1
                elif tmp.strip() == "":
                    tmp_prevcontext["Empty"] += 1
                else:
                    tmp_prevcontext["OK"] += 1
                model = ".".join(line["meta"].split(".")[:-1])
                segment_id = int(line["meta"].split(".")[-1].removeprefix("seg_"))
                score = line["annotations"][0]["quality_rating"]

                data[segment_id].append(
                    {
                        "model": model,
                        "score": score,
                        "tgt": line["target"],
                        "doc": line["meta"].split(".")[2],
                        "src": line["source"] if langs in {"en-zh", "en-de"} else "",
                    }
                )
            print(langs, "prevcontext", tmp_prevcontext)

            data_all[("iwslt25", langs)] = [
                {
                    "i": segment_id,
                    "doc": items[0]["doc"],
                    "src": items[0]["src"],
                    "tgt": {item["model"]: item["tgt"] for item in items},
                    "scores": {
                        item["model"]: {"human": item["score"]} for item in items
                    },
                }
                for segment_id, items in data.items()
            ]

    if compute_automated_metrics:
        import comet

        for metric_name in ["Unbabel/wmt22-cometkiwi-da", "zouharvi/COMET-partial"]:
            model = comet.load_from_checkpoint(comet.download_model(metric_name))

            for langs in ["en-zh", "en-de"]:
                srctgt_unique = list(
                    {
                        (line["src"], tgt)
                        for line in data_all[("iwslt25", langs)]
                        for tgt in line["tgt"].values()
                    }
                )
                srctgt_to_score = {
                    srctgt: score
                    for srctgt, score in zip(
                        srctgt_unique,
                        model.predict(
                            [{"src": src, "mt": tgt} for src, tgt in srctgt_unique],
                            batch_size=64,
                        ).scores,
                    )
                }
                for line in data_all[("iwslt25", langs)]:
                    for model, tgt in line["tgt"].items():
                        line["scores"][model][metric_name] = srctgt_to_score[
                            (line["src"], tgt)
                        ]

    return data_all
