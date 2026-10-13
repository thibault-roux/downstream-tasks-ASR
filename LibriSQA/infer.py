import csv
import json
import os
import re

from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


MODEL_NAME = "Qwen/Qwen3-8B"
FILE_PATH = "data/LibriSQA-Final.json"


def sanitize_model_name(name):
    return name.replace("/", "_")


def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=(
            torch.float16
            if torch.cuda.is_available()
            else torch.float32
        ),
        device_map="auto",
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=10,
        do_sample=False,
    )

    return pipe


def extract_answer(text):
    """
    Extract the first standalone A/B/C/D answer.

    This is slightly safer than searching for [ABCD] directly,
    because the model may theoretically generate other text.
    """
    match = re.search(r"\b([ABCD])\b", text.upper())

    if match:
        return match.group(1)

    return ""


def ask_llm(pipe, context, question_block):
    prompt = f"""You are a multiple-choice QA system.

Context:
{context}

{question_block}

Answer ONLY with one letter: A, B, C, or D.
"""

    output = pipe(prompt)[0]["generated_text"]

    # Usually generated_text contains prompt + completion.
    # Remove the prompt if it is present.
    if output.startswith(prompt):
        generated = output[len(prompt):].strip()
    else:
        generated = output.strip()

    return extract_answer(generated)


def load_log(log_path):
    """
    Load the inference cache.

    Returns:
        dict[text] = prediction
    """
    cache = {}

    if not os.path.exists(log_path):
        return cache

    with open(log_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            text = row["text"]
            prediction = row["prediction"]

            cache[text] = prediction

    return cache


def initialize_log(log_path):
    """
    Create the CSV file with its header if it does not already exist.
    """
    if os.path.exists(log_path):
        return

    with open(log_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["text", "prediction"],
        )
        writer.writeheader()


def append_to_log(log_path, text, prediction):
    """
    Append one inference result to the CSV cache immediately.
    """
    with open(log_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["text", "prediction"],
        )

        writer.writerow({
            "text": text,
            "prediction": prediction,
        })


def save_results(results_path, data):
    """
    Save the complete dataset, including all original fields and
    the newly added predictions.
    """
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False,
        )


def infer_text(
    pipe,
    text,
    question_block,
    cache,
    log_path,
):
    """
    Run inference for one text, using the CSV cache to avoid
    recomputing already processed texts.

    Returns the prediction.
    """

    # Already computed
    if text in cache:
        return cache[text]

    # New inference
    prediction = ask_llm(
        pipe,
        text,
        question_block,
    )

    # Update in-memory cache
    cache[text] = prediction

    # Persist immediately
    append_to_log(
        log_path,
        text,
        prediction,
    )

    return prediction


def evaluate_inference(
    data,
    pipe,
    results_path,
    log_path,
):
    """
    Add predictions to every hats entry.

    No evaluation is performed here.
    """

    cache = load_log(log_path)

    print(f"Loaded {len(cache)} cached text inferences.")

    # Count all potential inference items for progress reporting.
    total_inferences = sum(
        len(item.get("hats", [])) * 3
        for item in data
    )

    with tqdm(
        total=total_inferences,
        desc="Running inference",
    ) as pbar:

        for item in data:
            question_block = item["question_and_option"]

            for hat in item.get("hats", []):
                # -------------------------------------------------
                # Reference
                # -------------------------------------------------
                reference_text = hat["reference"]

                hat["ref_prediction"] = infer_text(
                    pipe=pipe,
                    text=reference_text,
                    question_block=question_block,
                    cache=cache,
                    log_path=log_path,
                )

                pbar.update(1)

                # -------------------------------------------------
                # H1
                # -------------------------------------------------
                h1_text = hat["h1_text"]

                hat["h1_prediction"] = infer_text(
                    pipe=pipe,
                    text=h1_text,
                    question_block=question_block,
                    cache=cache,
                    log_path=log_path,
                )

                pbar.update(1)

                # -------------------------------------------------
                # H2
                # -------------------------------------------------
                h2_text = hat["h2_text"]

                hat["h2_prediction"] = infer_text(
                    pipe=pipe,
                    text=h2_text,
                    question_block=question_block,
                    cache=cache,
                    log_path=log_path,
                )

                pbar.update(1)

            # Save the complete JSON after every sample.
            # This makes the run resumable even if it is interrupted.
            save_results(
                results_path,
                data,
            )

    # Final save
    save_results(
        results_path,
        data,
    )


def main():
    model_folder = sanitize_model_name(MODEL_NAME)

    output_dir = os.path.join(
        "answers",
        model_folder,
    )

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    results_path = os.path.join(
        output_dir,
        "LibriSQA-Results.json",
    )

    log_path = os.path.join(
        output_dir,
        "log.csv",
    )

    # -------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------
    data = load_data(FILE_PATH)

    # -------------------------------------------------------------
    # Initialize inference log
    # -------------------------------------------------------------
    initialize_log(log_path)

    # -------------------------------------------------------------
    # Load model
    # -------------------------------------------------------------
    pipe = load_model()

    # -------------------------------------------------------------
    # Run inference
    # -------------------------------------------------------------
    evaluate_inference(
        data=data,
        pipe=pipe,
        results_path=results_path,
        log_path=log_path,
    )

    print("\nInference complete.")
    print(f"Results saved to: {results_path}")
    print(f"Inference log saved to: {log_path}")


if __name__ == "__main__":
    main()