import json
import pickle as pk

data_dir = "./data/"
model_name = 'distilbert-base-uncased'
lang = 'en'

data = dict()
for segment_name in ['train', 'test', 'dev']:
    print(segment_name)
    raw_data = [json.loads(line) for line in open(data_dir + f'dataset_{lang}_{segment_name}.json', 'r', encoding='utf-8')]
    data[segment_name] = {
        "text": [r['review_body'] for r in raw_data],
        "original_labels": [int(r['stars']) for r in raw_data],
        "binary_labels": [0 if int(r['stars']) <= 3 else 1 for r in raw_data]
    }
    print("generating encoding...")
    pk.dump(data[segment_name], open(data_dir + f"{lang}_{segment_name}.pk", "wb"))
