
# in ../../../hats-en.csv, the format is:
# batch,triplet_id,reference,h1_text,h2_text,h1_votes,h2_votes
# 01,set_01_121-121726-0000.wav,also a popular contrivance whereby love making may be suspended but not stopped during the picnic season,also a popular condrivens whereby lovemaking may be suspended but not stopped during the technique season,also a popular contrivance where am i love making may be suspended but not stopped during the picnic season,0,3
# ...

# in ./LibriSQA-PartII-train.json, the format is:
# {"text": "he knew the silver fleece his and zora's must be ruined", "duration": 3.865, "question": "What did he know about the silver fleece belonging to himself and Zora?", "answer": "He knew that it must be ruined.", "speech_path": "LibriSpeech/test-clean/1995/1837/1995-1837-0000.wav"}


# open file
import json


def get_data(part, data_type):
    with open(f"LibriSQA-Part{part}-{data_type}.json", "r") as f:
        data = [json.loads(line) for line in f.readlines()][0]
    return data

def get_hats_data():
    with open("../../../hats-en.csv", "r") as f:
        lines = f.readlines()[1:] # skip header
    data = []
    for line in lines:
        batch, triplet_id, reference, h1_text, h2_text, h1_votes, h2_votes = line.strip().split(",")
        data.append({
            "batch": batch, # correspond to the path
            "triplet_id": triplet_id,
            "reference": reference,
            "h1_text": h1_text,
            "h2_text": h2_text,
            "h1_votes": int(h1_votes),
            "h2_votes": int(h2_votes)
        })
    return data

# check if wav contained in data exists in hats
def check_wav(data, hats_data):
    # take path from triplet_id column, split("_")[2]
    not_found = 0
    found = 0
    not_found_wavs = []
    found_wavs = []
    wavs = set()
    for item in hats_data:
        wavs.add(item["triplet_id"].split("_")[2])

    new_data = []
    for item in data:
        wav_path = item["speech_path"].split("/")[-1]
        if wav_path not in wavs:
            not_found += 1
            not_found_wavs.append(wav_path)
            # print(f"WAV file {wav_path} not found in HATS data")
        else:
            found += 1
            found_wavs.append(wav_path)
            new_data.append(item)

    # print(set(not_found_wavs[:100]))
    # print("^^^ not found wavs ^^^")
    # input("Press Enter to continue...")
    # print(set(found_wavs[:100]))
    # print("^^^ found wavs ^^^")
    # input("Press Enter to continue...")
    # print(wavs)
    # print("^^^ all wavs in HATS ^^^")
    # input("Press Enter to continue...")

    print(f"\tnot found in HATS data: {not_found}")
    print(f"\tfound in HATS data: {found}")
    print(f"\tunique wavs in HATS data: {len(wavs)}")

    # save found data to json
    with open("LibriSQA-Found.json", "w") as f:
        json.dump(new_data, f, indent=4)


# if __name__ == "__main__":
#     hats_data = get_hats_data()
#     print(f"samples in HATS: {len(hats_data)}")

#     data_types = ["train", "test"]
#     parts = ["I", "II"]
#     for data_type in data_types:
#         for part in parts:
#             print(f"Checking LibriSQA Part {part} {data_type}...")
#             data = get_data(part, data_type)
#             print(f"\tsamples in LibriSQA Part : {len(data)}")
            
#             check_wav(data, hats_data)

if __name__ == "__main__":
    hats_data = get_hats_data()
    print(f"samples in HATS: {len(hats_data)}")

    data_types = ["train", "test"]
    parts = ["I", "II"]
    concat_data = []
    for data_type in data_types:
        for part in parts:
            data = get_data(part, data_type)
            concat_data.extend(data)

    print(f"Total samples in LibriSQA: {len(concat_data)}")

    check_wav(concat_data, hats_data)