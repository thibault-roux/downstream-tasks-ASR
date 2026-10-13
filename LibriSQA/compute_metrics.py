#!/usr/bin/env python3

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Tuple

from jiwer import wer, cer


# Metrics currently enabled.
#
# When adding BERTScore or semantic distance back, add them here
# and uncomment their computation below.
ACTIVE_METRICS = [
    "wer",
    "cer",
]


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_metric_cache(path: Path) -> Dict[Tuple[str, str], float]:
    """
    Load a metric cache of the form:

        reference,hypothesis,score

    into:

        {(reference, hypothesis): score}
    """
    cache = {}

    if not path.exists():
        return cache

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            reference = row["reference"]
            hypothesis = row["hypothesis"]
            score = float(row["score"])

            cache[(reference, hypothesis)] = score

    return cache


def save_metric_cache(
    path: Path,
    cache: Dict[Tuple[str, str], float],
):
    path.parent.mkdir(parents=True, exist_ok=True)

    items = sorted(
        cache.items(),
        key=lambda x: (x[0][0], x[0][1]),
    )

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "reference",
            "hypothesis",
            "score",
        ])

        for (reference, hypothesis), score in items:
            writer.writerow([
                reference,
                hypothesis,
                score,
            ])


def collect_pairs(data):
    """
    Collect all unique (reference, hypothesis) pairs appearing
    in the HATS triplets.
    """
    pairs = set()

    for item in data:
        for hat in item.get("hats", []):
            reference = hat["reference"]
            h1 = hat["h1_text"]
            h2 = hat["h2_text"]

            pairs.add((reference, h1))
            pairs.add((reference, h2))

    return pairs


def compute_wer_cer(
    pairs,
    wer_cache,
    cer_cache,
):
    """
    Compute WER and CER for missing pairs only.
    """

    missing_wer = [
        (reference, hypothesis)
        for reference, hypothesis in pairs
        if (reference, hypothesis) not in wer_cache
    ]

    missing_cer = [
        (reference, hypothesis)
        for reference, hypothesis in pairs
        if (reference, hypothesis) not in cer_cache
    ]

    print(f"WER: computing {len(missing_wer)} new pairs")
    print(f"CER: computing {len(missing_cer)} new pairs")

    for reference, hypothesis in missing_wer:
        wer_cache[(reference, hypothesis)] = wer(
            reference,
            hypothesis,
        )

    for reference, hypothesis in missing_cer:
        cer_cache[(reference, hypothesis)] = cer(
            reference,
            hypothesis,
        )


def choose_best(
    h1_score,
    h2_score,
    lower_is_better,
):
    """
    Return "h1" or "h2".

    For WER/CER:
        lower is better.

    For similarity metrics such as BERTScore:
        higher is better.

    Ties are resolved in favor of h1.
    """
    if lower_is_better:
        return "h1" if h1_score <= h2_score else "h2"

    return "h1" if h1_score >= h2_score else "h2"


def add_metric_winners(
    data,
    caches,
):
    """
    Add the winning hypothesis for every active metric.
    """

    for item in data:
        for hat in item.get("hats", []):
            reference = hat["reference"]
            h1 = hat["h1_text"]
            h2 = hat["h2_text"]

            if "wer" in ACTIVE_METRICS:
                wer_h1 = caches["wer"][
                    (reference, h1)
                ]
                wer_h2 = caches["wer"][
                    (reference, h2)
                ]

                hat["wer"] = choose_best(
                    wer_h1,
                    wer_h2,
                    lower_is_better=True,
                )

            if "cer" in ACTIVE_METRICS:
                cer_h1 = caches["cer"][
                    (reference, h1)
                ]
                cer_h2 = caches["cer"][
                    (reference, h2)
                ]

                hat["cer"] = choose_best(
                    cer_h1,
                    cer_h2,
                    lower_is_better=True,
                )

            # Uncomment when BERTScore is enabled:
            #
            # if "bertscore" in ACTIVE_METRICS:
            #     bertscore_h1 = caches["bertscore"][
            #         (reference, h1)
            #     ]
            #     bertscore_h2 = caches["bertscore"][
            #         (reference, h2)
            #     ]
            #
            #     hat["bertscore"] = choose_best(
            #         bertscore_h1,
            #         bertscore_h2,
            #         lower_is_better=False,
            #     )

            # Uncomment when semantic distance is enabled:
            #
            # if "semdist" in ACTIVE_METRICS:
            #     semdist_h1 = caches["semdist"][
            #         (reference, h1)
            #     ]
            #     semdist_h2 = caches["semdist"][
            #         (reference, h2)
            #     ]
            #
            #     hat["semdist"] = choose_best(
            #         semdist_h1,
            #         semdist_h2,
            #         lower_is_better=True,
            #     )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute WER/CER for LibriSQA HATS triplets."
        )
    )

    parser.add_argument(
        "--llm",
        default="Qwen/Qwen3-8B",
        help=(
            "LLM name. '/' is automatically replaced by '_'. "
            "Default: Qwen/Qwen3-8B"
        ),
    )

    parser.add_argument(
        "--answers-dir",
        default="answers",
        help="Base answers directory. Default: answers",
    )

    parser.add_argument(
        "--metrics-dir",
        default="metrics_scores",
        help="Metric cache directory. Default: metrics_scores",
    )

    args = parser.parse_args()

    # Automatically convert e.g.
    #
    # Qwen/Qwen3-8B
    #
    # into:
    #
    # Qwen_Qwen3-8B
    llm = args.llm.replace("/", "_")

    answers_dir = Path(args.answers_dir)
    metrics_dir = Path(args.metrics_dir)

    input_path = (
        answers_dir
        / llm
        / "LibriSQA-Results.json"
    )

    output_path = (
        answers_dir
        / llm
        / "LibriSQA-Results-Metrics.json"
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file does not exist: {input_path}"
        )

    print(f"LLM: {args.llm}")
    print(f"LLM directory: {llm}")
    print(f"Input: {input_path}")
    print(f"Active metrics: {', '.join(ACTIVE_METRICS)}")

    data = load_json(input_path)

    if not isinstance(data, list):
        raise ValueError(
            "Expected the input JSON to contain a list."
        )

    pairs = collect_pairs(data)

    print(
        f"Found {len(pairs)} unique "
        "reference/hypothesis pairs."
    )

    # --------------------------------------------------------------
    # Load caches only for active metrics
    # --------------------------------------------------------------

    caches = {}

    for metric in ACTIVE_METRICS:
        cache_path = metrics_dir / f"{metric}.csv"

        caches[metric] = load_metric_cache(
            cache_path
        )

        print(
            f"{metric}: "
            f"{len(caches[metric])} cached scores "
            f"from {cache_path}"
        )

    # --------------------------------------------------------------
    # Compute active metrics
    # --------------------------------------------------------------

    if "wer" in ACTIVE_METRICS or "cer" in ACTIVE_METRICS:
        # The function can compute both. We only pass the caches
        # that are actually active.
        #
        # If you later disable one of these, this function can
        # easily be split into two independent functions.
        if "wer" not in caches:
            caches["wer"] = {}

        if "cer" not in caches:
            caches["cer"] = {}

        compute_wer_cer(
            pairs,
            caches["wer"],
            caches["cer"],
        )

    # --------------------------------------------------------------
    # BERTScore intentionally disabled for now.
    # --------------------------------------------------------------
    #
    # if "bertscore" in ACTIVE_METRICS:
    #     compute_bertscore(...)

    # --------------------------------------------------------------
    # Semantic distance intentionally disabled for now.
    # --------------------------------------------------------------
    #
    # if "semdist" in ACTIVE_METRICS:
    #     compute_semdist(...)

    # --------------------------------------------------------------
    # Save metric caches
    # --------------------------------------------------------------

    for metric in ACTIVE_METRICS:
        cache_path = metrics_dir / f"{metric}.csv"

        save_metric_cache(
            cache_path,
            caches[metric],
        )

        print(
            f"Saved {metric} cache: {cache_path}"
        )

    # --------------------------------------------------------------
    # Add winners to the JSON
    # --------------------------------------------------------------

    add_metric_winners(
        data,
        caches,
    )

    save_json(
        output_path,
        data,
    )

    print(f"Saved results: {output_path}")


if __name__ == "__main__":
    main()