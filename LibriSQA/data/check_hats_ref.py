#!/usr/bin/env python3

import csv
from collections import defaultdict
from pathlib import Path

CSV_FILE = Path("../../../hats-en.csv")

# Real WAV filename -> set of references associated with it
wav_to_references = defaultdict(set)

with CSV_FILE.open("r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)

    for row in reader:
        triplet_id = row["triplet_id"]

        # Remove the "set_XX_" prefix:
        # set_01_1320-122612-0016.wav -> 1320-122612-0016.wav
        wav_file = triplet_id.split("_", 2)[-1]

        reference = row["reference"]
        wav_to_references[wav_file].add(reference)

# Find WAV files associated with multiple unique references
conflicts = {
    wav_file: references
    for wav_file, references in wav_to_references.items()
    if len(references) > 1
}

if not conflicts:
    print("OK: every WAV file is associated with exactly one unique reference.")
else:
    print(f"ERROR: {len(conflicts)} WAV files have multiple references:\n")

    for wav_file, references in sorted(conflicts.items()):
        print(f"{wav_file}:")
        for reference in sorted(references):
            print(f"  - {reference}")

print(f"\nTotal unique WAV files: {len(wav_to_references)}")
print(f"WAV files with conflicting references: {len(conflicts)}")