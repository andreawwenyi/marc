import json
import pickle as pk
import os

data_dir = "./data/"

for filename in os.listdir(data_dir):
    if filename.endswith("json"):
        print(filename)
        _, lang, segment = filename.strip(".json").split("_")
        raw_data = [json.loads(line) for line in open(data_dir + filename,
                                                      'r',
                                                      encoding='utf-8')]
        processed = {
            "text": [r['review_body'] for r in raw_data],
            "original_labels": [int(r['stars']) for r in raw_data],
            "binary_labels": [0 if int(r['stars']) <= 3 else 1 for r in raw_data]
        }
        output_filename = f"clean_{lang}_{segment}.pk"
        pk.dump(processed, open(data_dir + output_filename, "wb"))
