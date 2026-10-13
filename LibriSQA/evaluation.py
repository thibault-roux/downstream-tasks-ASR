import os
import json
from collections import defaultdict

import numpy as np
from sklearn.metrics import confusion_matrix


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

MODEL_NAME = "Qwen/Qwen3-8B"

RESULTS_FILENAME = "LibriSQA-Results.json"


def sanitize_model_name(name):
    return name.replace("/", "_")


# ------------------------------------------------------------------
# Loading
# ------------------------------------------------------------------

def load_results(results_path):
    with open(results_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------------
# Confusion matrix
# ------------------------------------------------------------------

def save_confusion_matrix(
    output_path,
    matrix,
    labels=("A", "B", "C", "D"),
    system_name=None,
):
    """
    Save a confusion matrix as a readable text file.

    Rows    = ground truth
    Columns = prediction
    """

    with open(output_path, "w", encoding="utf-8") as f:
        if system_name is not None:
            f.write(f"System: {system_name}\n\n")

        f.write("Rows = ground truth\n")
        f.write("Columns = prediction\n\n")

        # Header
        f.write(f"{'':>12}")
        for label in labels:
            f.write(f"{label:>10}")
        f.write("\n")

        # Rows
        for i, label in enumerate(labels):
            f.write(f"{label:>12}")
            for j in range(len(labels)):
                f.write(f"{matrix[i, j]:>10}")
            f.write("\n")


def print_confusion_matrix(
    matrix,
    labels=("A", "B", "C", "D"),
    system_name=None,
):
    """
    Print a confusion matrix to stdout.
    """

    if system_name:
        print(f"\nConfusion matrix: {system_name}")

    print("Rows = ground truth, Columns = prediction")
    print()

    print(f"{'':>12}", end="")
    for label in labels:
        print(f"{label:>10}", end="")
    print()

    for i, label in enumerate(labels):
        print(f"{label:>12}", end="")
        for j in range(len(labels)):
            print(f"{matrix[i, j]:>10}", end="")
        print()


# ------------------------------------------------------------------
# System evaluation
# ------------------------------------------------------------------

def collect_system_predictions(data):
    """
    Collect predictions by actual ASR system.

    Each hat contains:

        model1
        model2
        h1_text
        h2_text
        h1_prediction
        h2_prediction

    model1 is the ASR system corresponding to h1_text/h1_prediction.
    model2 is the ASR system corresponding to h2_text/h2_prediction.

    Returns:

        {
            "reference": {
                "ground_truth": [...],
                "prediction": [...]
            },

            "wav2vec-ft-asr": {
                "ground_truth": [...],
                "prediction": [...]
            },

            "googlesr": {
                "ground_truth": [...],
                "prediction": [...]
            }
        }
    """

    systems = defaultdict(
        lambda: {
            "ground_truth": [],
            "prediction": [],
        }
    )

    for item in data:
        answer = item["answer"]

        for hat in item.get("hats", []):

            # ------------------------------------------------------
            # Reference
            # ------------------------------------------------------

            if "ref_prediction" in hat:
                systems["reference"]["ground_truth"].append(answer)
                systems["reference"]["prediction"].append(
                    hat["ref_prediction"]
                )

            # ------------------------------------------------------
            # H1
            # ------------------------------------------------------

            if (
                "model1" in hat
                and "h1_prediction" in hat
            ):
                system = hat["model1"]

                systems[system]["ground_truth"].append(answer)
                systems[system]["prediction"].append(
                    hat["h1_prediction"]
                )

            # ------------------------------------------------------
            # H2
            # ------------------------------------------------------

            if (
                "model2" in hat
                and "h2_prediction" in hat
            ):
                system = hat["model2"]

                systems[system]["ground_truth"].append(answer)
                systems[system]["prediction"].append(
                    hat["h2_prediction"]
                )

    return systems


def evaluate_systems(data, output_dir):
    """
    Compute precision/accuracy and confusion matrices.

    For this single-label multiple-choice task, the requested
    'precision' is computed as:

        correct predictions / all predictions

    This is equivalent to accuracy.
    """

    systems = collect_system_predictions(data)

    confusion_dir = os.path.join(
        output_dir,
        "confusion_matrix",
    )

    os.makedirs(confusion_dir, exist_ok=True)

    scores = {}

    print("\n" + "=" * 70)
    print("SYSTEM EVALUATION")
    print("=" * 70)

    for system_name, values in systems.items():

        ground_truth = values["ground_truth"]
        predictions = values["prediction"]

        if not ground_truth:
            continue

        correct = sum(
            pred == gt
            for pred, gt in zip(predictions, ground_truth)
        )

        total = len(ground_truth)

        precision = correct / total

        scores[system_name] = {
            "precision": precision,
            "correct": correct,
            "total": total,
        }

        print(f"\n{system_name}")
        print("-" * 70)
        print(f"Correct:   {correct}/{total}")
        print(f"Precision: {precision:.6f} ({precision * 100:.2f}%)")

        # ----------------------------------------------------------
        # Confusion matrix
        # ----------------------------------------------------------

        labels = ["A", "B", "C", "D"]

        matrix = confusion_matrix(
            ground_truth,
            predictions,
            labels=labels,
        )

        print_confusion_matrix(
            matrix,
            labels=labels,
            system_name=system_name,
        )

        # # Save separate matrix for this system
        # system_dir = os.path.join(
        #     confusion_dir,
        #     system_name,
        # )

        # os.makedirs(
        #     system_dir,
        #     exist_ok=True,
        # )

        matrix_path = os.path.join(
            confusion_dir, # system_dir,
            f"{system_name}.txt",
        )

        save_confusion_matrix(
            matrix_path,
            matrix,
            labels=labels,
            system_name=system_name,
        )

    # --------------------------------------------------------------
    # Save scores
    # --------------------------------------------------------------

    scores_path = os.path.join(
        output_dir,
        "scores_systems.txt",
    )

    with open(scores_path, "w", encoding="utf-8") as f:
        f.write("LibriSQA system scores\n")
        f.write("=" * 60 + "\n\n")

        for system_name, score in scores.items():
            f.write(
                f"{system_name}: "
                f"{score['precision']:.6f} "
                f"({score['precision'] * 100:.2f}%)\n"
            )

            f.write(
                f"Correct: {score['correct']}/{score['total']}\n\n"
            )

    print(f"\nScores saved to: {scores_path}")


# ------------------------------------------------------------------
# Human-choice analysis
# ------------------------------------------------------------------

def human_choice_analysis(data, output_dir):
    """
    Measure how human selection affects transcript quality.

    Four categories:

    1. both_correct
       h1_prediction == answer == h2_prediction

    2. both_incorrect
       h1_prediction != answer AND h2_prediction != answer

    3. humans_wrong
       Humans selected the incorrect hypothesis:
           h1_votes > h2_votes AND
           h1_prediction != answer AND
           h2_prediction == answer

       OR:

           h2_votes > h1_votes AND
           h2_prediction != answer AND
           h1_prediction == answer

    4. humans_right
       Humans selected the correct hypothesis:
           h1_votes > h2_votes AND
           h1_prediction == answer AND
           h2_prediction != answer

       OR:

           h2_votes > h1_votes AND
           h2_prediction == answer AND
           h1_prediction != answer

    Cases where h1_votes == h2_votes are recorded separately as
    "tie", because there is no human preference.
    """

    categories = {
        "both_correct": 0.0,
        "both_incorrect": 0.0,
        "humans_wrong": 0.0,
        "humans_right": 0.0,
        "human_vote_tie": 0.0,
    }

    counts = {
        "both_correct": 0,
        "both_incorrect": 0,
        "humans_wrong": 0,
        "humans_right": 0,
        "human_vote_tie": 0,
    }

    total_duration = 0.0

    for item in data:

        answer = item["answer"]

        # Duration associated with this audio/question.
        duration = float(item.get("duration", 0.0))

        for hat in item.get("hats", []):

            h1_prediction = hat.get("h1_prediction", "")
            h2_prediction = hat.get("h2_prediction", "")

            h1_votes = hat.get("h1_votes", 0)
            h2_votes = hat.get("h2_votes", 0)

            # Skip incomplete predictions.
            if not h1_prediction or not h2_prediction:
                continue

            total_duration += duration

            # ------------------------------------------------------
            # Both hypotheses are correct
            # ------------------------------------------------------

            if (
                h1_prediction == answer
                and h2_prediction == answer
            ):
                categories["both_correct"] += duration
                counts["both_correct"] += 1

            # ------------------------------------------------------
            # Both hypotheses are incorrect
            # ------------------------------------------------------

            elif (
                h1_prediction != answer
                and h2_prediction != answer
            ):
                categories["both_incorrect"] += duration
                counts["both_incorrect"] += 1

            # ------------------------------------------------------
            # Humans selected the wrong hypothesis
            # ------------------------------------------------------

            elif (
                h1_votes > h2_votes
                and h1_prediction != answer
                and h2_prediction == answer
            ) or (
                h2_votes > h1_votes
                and h2_prediction != answer
                and h1_prediction == answer
            ):
                categories["humans_wrong"] += duration
                counts["humans_wrong"] += 1

            # ------------------------------------------------------
            # Humans selected the correct hypothesis
            # ------------------------------------------------------

            elif (
                h1_votes > h2_votes
                and h1_prediction == answer
                and h2_prediction != answer
            ) or (
                h2_votes > h1_votes
                and h2_prediction == answer
                and h1_prediction != answer
            ):
                categories["humans_right"] += duration
                counts["humans_right"] += 1

            # ------------------------------------------------------
            # Equal votes
            # ------------------------------------------------------

            elif h1_votes == h2_votes:
                categories["human_vote_tie"] += duration
                counts["human_vote_tie"] += 1

    # --------------------------------------------------------------
    # Percentages
    # --------------------------------------------------------------

    percentages = {}

    for category, duration in categories.items():
        if total_duration > 0:
            percentages[category] = (
                duration / total_duration * 100
            )
        else:
            percentages[category] = 0.0

    # --------------------------------------------------------------
    # Human improvement score
    # --------------------------------------------------------------
    #
    # This is the amount of time where human selection picked
    # the better transcript, minus the amount where humans picked
    # the worse transcript.
    #
    # Positive => humans improved transcript quality.
    # Negative => humans hurt transcript quality.
    #

    human_net_gain = (
        categories["humans_right"]
        - categories["humans_wrong"]
    )

    if total_duration > 0:
        human_net_gain_percentage = (
            human_net_gain / total_duration * 100
        )
    else:
        human_net_gain_percentage = 0.0

    # --------------------------------------------------------------
    # Save report
    # --------------------------------------------------------------

    output_path = os.path.join(
        output_dir,
        "human_scores.txt",
    )

    with open(output_path, "w", encoding="utf-8") as f:

        f.write("LibriSQA human transcript-selection analysis\n")
        f.write("=" * 70 + "\n\n")

        f.write(
            "Duration is accumulated per human comparison (hat).\n"
        )

        f.write(
            "Therefore, if an audio sample has two hat entries, "
            "its duration can contribute twice.\n\n"
        )

        f.write(
            f"Total comparison duration: "
            f"{total_duration:.3f} seconds\n"
        )

        f.write(
            f"Total comparison duration: "
            f"{total_duration / 3600:.3f} hours\n\n"
        )

        # ----------------------------------------------------------
        # Categories
        # ----------------------------------------------------------

        category_names = {
            "both_correct": "Both hypotheses correct",
            "both_incorrect": "Both hypotheses incorrect",
            "humans_right": "Humans chose the right hypothesis",
            "humans_wrong": "Humans chose the wrong hypothesis",
            "human_vote_tie": "Human vote tie",
        }

        for category, name in category_names.items():

            duration = categories[category]
            percentage = percentages[category]
            count = counts[category]

            f.write(f"{name}\n")
            f.write("-" * 70 + "\n")
            f.write(f"Count:      {count}\n")
            f.write(
                f"Duration:   {duration:.3f} seconds\n"
            )
            f.write(
                f"Duration:   {duration / 3600:.3f} hours\n"
            )
            f.write(
                f"Percentage: {percentage:.2f}%\n\n"
            )

        # ----------------------------------------------------------
        # Main human-selection score
        # ----------------------------------------------------------

        f.write("Human selection effect\n")
        f.write("-" * 70 + "\n")

        f.write(
            "Humans chose right duration: "
            f"{categories['humans_right']:.3f} seconds\n"
        )

        f.write(
            "Humans chose wrong duration: "
            f"{categories['humans_wrong']:.3f} seconds\n"
        )

        f.write(
            "Net human gain: "
            f"{human_net_gain:.3f} seconds\n"
        )

        f.write(
            "Net human gain: "
            f"{human_net_gain / 3600:.3f} hours\n"
        )

        f.write(
            "Net human gain percentage: "
            f"{human_net_gain_percentage:.2f}%\n"
        )

    # --------------------------------------------------------------
    # Print report
    # --------------------------------------------------------------

    print("\n" + "=" * 70)
    print("HUMAN TRANSCRIPT-SELECTION ANALYSIS")
    print("=" * 70)

    print(
        f"\nTotal comparison duration: "
        f"{total_duration:.3f} seconds"
    )

    for category, name in category_names.items():
        print(
            f"\n{name}:"
            f"\n  Count:      {counts[category]}"
            f"\n  Duration:   {categories[category]:.3f} s"
            f"\n  Percentage: {percentages[category]:.2f}%"
        )

    print("\nHuman selection effect:")
    print(
        f"  Humans chose right: "
        f"{categories['humans_right']:.3f} s"
    )
    print(
        f"  Humans chose wrong: "
        f"{categories['humans_wrong']:.3f} s"
    )
    print(
        f"  Net human gain:     "
        f"{human_net_gain:.3f} s"
        f" ({human_net_gain_percentage:.2f}%)"
    )

    print(f"\nHuman scores saved to: {output_path}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():

    model_folder = sanitize_model_name(MODEL_NAME)

    output_dir = os.path.join(
        "answers",
        model_folder,
    )

    results_path = os.path.join(
        output_dir,
        RESULTS_FILENAME,
    )

    if not os.path.exists(results_path):
        raise FileNotFoundError(
            f"Results file not found: {results_path}"
        )

    data = load_results(results_path)

    print(f"Loaded {len(data)} samples from:")
    print(results_path)

    # --------------------------------------------------------------
    # 1. System scores + confusion matrices
    # --------------------------------------------------------------

    evaluate_systems(
        data,
        output_dir,
    )

    # --------------------------------------------------------------
    # 2. Human selection analysis
    # --------------------------------------------------------------

    human_choice_analysis(
        data,
        output_dir,
    )


if __name__ == "__main__":
    main()