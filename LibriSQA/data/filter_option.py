

# open ./LibriSQA-Found.json
# only keep the samples that contains "question_and_option"
import json

def read_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data

if __name__ == "__main__":
    data = read_json("LibriSQA-Found.json")
    print(f"Total samples in LibriSQA-Found: {len(data)}")

    new_data = []
    for item in data:
        if "question_and_option" in item:
            new_data.append(item)

    print(f"Total samples with question_and_option: {len(new_data)}")

    # save new data to json
    with open("LibriSQA-Filtered.json", "w") as f:
        json.dump(new_data, f, indent=4)