#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


METRICS = [
    "wer",
    "cer",
    # "bertscore",
    # "semdist",
]


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def compute_accuracy(data, metric):
    """
    For each HATS triplet:

      - metric tells us whether h1 or h2 is the best hypothesis
      - h1_prediction / h2_prediction are the predictions made from
        the corresponding hypotheses
      - answer is the ground-truth answer

    The metric is considered correct if it selects a hypothesis whose
    prediction matches the ground truth.

    If both h1 and h2 have the same correctness with respect to the
    ground truth, the metric is counted as correct only when its choice
    is one of the correct hypotheses.

    Returns:
        {
            "total": ...,
            "metric_correct": ...,
            "metric_accuracy": ...,
            "h1_correct": ...,
            "h2_correct": ...,
            "both_correct": ...,
            "both_wrong": ...,
        }
    """

    total = 0
    metric_correct = 0

    h1_correct = 0
    h2_correct = 0

    both_correct = 0
    both_wrong = 0

    for item in data:
        answer = item["answer"]

        for hat in item.get("hats", []):
            metric_choice = hat.get(metric)

            h1_prediction = hat.get("h1_prediction")
            h2_prediction = hat.get("h2_prediction")

            if metric_choice not in {"h1", "h2"}:
                continue

            if h1_prediction is None or h2_prediction is None:
                continue

            total += 1

            h1_is_correct = h1_prediction == answer
            h2_is_correct = h2_prediction == answer

            if h1_is_correct:
                h1_correct += 1

            if h2_is_correct:
                h2_correct += 1

            if h1_is_correct and h2_is_correct:
                both_correct += 1

            if not h1_is_correct and not h2_is_correct:
                both_wrong += 1

            # The metric selected h1/h2. Check whether the selected
            # hypothesis produced the correct answer.
            selected_is_correct = (
                h1_is_correct
                if metric_choice == "h1"
                else h2_is_correct
            )

            if selected_is_correct:
                metric_correct += 1

    accuracy = (
        metric_correct / total
        if total > 0
        else 0.0
    )

    return {
        "total": total,
        "metric_correct": metric_correct,
        "metric_accuracy": accuracy,
        "h1_correct": h1_correct,
        "h2_correct": h2_correct,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
    }


def compute_metric_selection_stats(data, metric):
    """
    Additional statistics describing which hypothesis the metric
    selects and how often that selection is actually useful.

    In particular:

      - selected_h1 / selected_h2
      - selected_correct
      - selected_wrong
      - oracle_correct

    oracle_correct = number of examples where at least one of h1/h2
    gives the correct answer.

    This allows us to distinguish metric quality from the intrinsic
    difficulty of the HATS pair.
    """

    selected_h1 = 0
    selected_h2 = 0

    selected_correct = 0
    selected_wrong = 0

    oracle_correct = 0

    for item in data:
        answer = item["answer"]

        for hat in item.get("hats", []):
            choice = hat.get(metric)

            if choice not in {"h1", "h2"}:
                continue

            h1_prediction = hat.get("h1_prediction")
            h2_prediction = hat.get("h2_prediction")

            if h1_prediction is None or h2_prediction is None:
                continue

            h1_correct = h1_prediction == answer
            h2_correct = h2_prediction == answer

            if h1_correct or h2_correct:
                oracle_correct += 1

            if choice == "h1":
                selected_h1 += 1
                selected_is_correct = h1_correct
            else:
                selected_h2 += 1
                selected_is_correct = h2_correct

            if selected_is_correct:
                selected_correct += 1
            else:
                selected_wrong += 1

    total = selected_h1 + selected_h2

    selection_rate_h1 = (
        selected_h1 / total
        if total
        else 0.0
    )

    selection_rate_h2 = (
        selected_h2 / total
        if total
        else 0.0
    )

    oracle_accuracy = (
        oracle_correct / total
        if total
        else 0.0
    )

    return {
        "selected_h1": selected_h1,
        "selected_h2": selected_h2,
        "selection_rate_h1": selection_rate_h1,
        "selection_rate_h2": selection_rate_h2,
        "selected_correct": selected_correct,
        "selected_wrong": selected_wrong,
        "oracle_correct": oracle_correct,
        "oracle_accuracy": oracle_accuracy,
    }


def format_results(metric, stats, selection_stats):
    lines = []

    lines.append(f"Metric: {metric}")
    lines.append("-" * 60)

    lines.append(
        f"Total evaluated:       {stats['total']}"
    )

    lines.append(
        f"Metric correct:        {stats['metric_correct']}"
    )

    lines.append(
        f"Metric accuracy:       "
        f"{stats['metric_accuracy'] * 100:.2f}%"
    )

    lines.append("")

    lines.append(
        f"H1 correct:            {stats['h1_correct']}"
    )

    lines.append(
        f"H2 correct:            {stats['h2_correct']}"
    )

    lines.append(
        f"Both correct:          {stats['both_correct']}"
    )

    lines.append(
        f"Both wrong:            {stats['both_wrong']}"
    )

    lines.append("")

    lines.append(
        f"Selected H1:           "
        f"{selection_stats['selected_h1']} "
        f"({selection_stats['selection_rate_h1'] * 100:.2f}%)"
    )

    lines.append(
        f"Selected H2:           "
        f"{selection_stats['selected_h2']} "
        f"({selection_stats['selection_rate_h2'] * 100:.2f}%)"
    )

    lines.append(
        f"Selected correct:      "
        f"{selection_stats['selected_correct']}"
    )

    lines.append(
        f"Selected wrong:        "
        f"{selection_stats['selected_wrong']}"
    )

    lines.append("")

    lines.append(
        f"Oracle correct:        "
        f"{selection_stats['oracle_correct']}"
    )

    lines.append(
        f"Oracle accuracy:       "
        f"{selection_stats['oracle_accuracy'] * 100:.2f}%"
    )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate how well ASR metrics select the hypothesis "
            "that produces the correct LibriSQA answer."
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

    args = parser.parse_args()

    # Convert e.g.
    #
    # Qwen/Qwen3-8B
    #
    # to:
    #
    # Qwen_Qwen3-8B
    llm = args.llm.replace("/", "_")

    answers_dir = Path(args.answers_dir)

    input_path = (
        answers_dir
        / llm
        / "LibriSQA-Results-Metrics.json"
    )

    output_path = (
        answers_dir
        / llm
        / "scores_metrics.txt"
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file does not exist: {input_path}"
        )

    print(f"LLM: {args.llm}")
    print(f"LLM directory: {llm}")
    print(f"Input: {input_path}")

    data = load_json(input_path)

    if not isinstance(data, list):
        raise ValueError(
            "Expected the input JSON to contain a list."
        )

    output_lines = []

    output_lines.append(
        f"LibriSQA metric evaluation"
    )
    output_lines.append(
        f"LLM: {args.llm}"
    )
    output_lines.append(
        f"Input: {input_path}"
    )
    output_lines.append("")
    output_lines.append("=" * 60)
    output_lines.append("")

    for metric in METRICS:
        stats = compute_accuracy(
            data,
            metric,
        )

        selection_stats = compute_metric_selection_stats(
            data,
            metric,
        )

        result = format_results(
            metric,
            stats,
            selection_stats,
        )

        print()
        print(result)

        output_lines.append(result)
        output_lines.append("")
        output_lines.append("=" * 60)
        output_lines.append("")

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write("\n".join(output_lines))

    print()
    print(f"Saved results: {output_path}")


if __name__ == "__main__":
    main()
