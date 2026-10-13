import csv
import json
import os
import re


FOUND_FILE = "LibriSQA-Filtered.json"
FINAL_FILE = "LibriSQA-Final.json"
HATS_FILE = "../../../hats-en-systems.csv"


def get_wav_filename(path):
    """Return only the wav filename from a path."""
    return os.path.basename(path)


def get_hats_wav_filename(triplet_id):
    """
    Convert e.g.
        set_01_1320-122612-0016.wav
    to:
        1320-122612-0016.wav
    """
    return re.sub(r"^set_[^_]+_", "", triplet_id)


def load_hats(hats_file):
    """
    Load HATS data and group rows by wav filename.

    Multiple HATS rows can correspond to the same wav, so the
    result is:
        {
            "1320-122612-0016.wav": [
                {...},
                {...}
            ]
        }
    """
    hats_by_wav = {}

    with open(hats_file, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            wav_filename = get_hats_wav_filename(row["triplet_id"])

            hats_entry = {
                "reference": row["reference"],
                "h1_text": row["h1_text"],
                "h2_text": row["h2_text"],
                "h1_votes": int(row["h1_votes"]),
                "h2_votes": int(row["h2_votes"]),
                "model1": row["h1_system"],
                "model2": row["h2_system"],
            }

            hats_by_wav.setdefault(wav_filename, []).append(hats_entry)

    return hats_by_wav


def main():
    # If the final file already exists, do nothing.
    if os.path.exists(FINAL_FILE):
        print(f"{FINAL_FILE} already exists. Nothing to do.")
        return

    # Load the original LibriSQA data.
    if not os.path.exists(FOUND_FILE):
        raise FileNotFoundError(
            f"Could not find {FOUND_FILE}"
        )

    with open(FOUND_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Load HATS data.
    if not os.path.exists(HATS_FILE):
        raise FileNotFoundError(
            f"Could not find {HATS_FILE}"
        )

    hats_by_wav = load_hats(HATS_FILE)

    # Add HATS information to every LibriSQA entry.
    matched = 0
    unmatched = 0

    for item in data:
        speech_path = item.get("speech_path")

        if not speech_path:
            item["hats"] = []
            unmatched += 1
            continue

        wav_filename = get_wav_filename(speech_path)

        if wav_filename in hats_by_wav:
            item["hats"] = hats_by_wav[wav_filename]
            matched += 1
        else:
            item["hats"] = []
            unmatched += 1

    # Save the final file.
    with open(FINAL_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"Created {FINAL_FILE}")
    print(f"Matched: {matched}")
    print(f"Unmatched: {unmatched}")


if __name__ == "__main__":
    main()
