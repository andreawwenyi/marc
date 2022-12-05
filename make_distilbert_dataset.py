import json
import pickle as pk

import torch
from transformers import DistilBertTokenizerFast

data_dir = "./data/"
model_name = 'distilbert-base-uncased'
lang = 'en'

# load the encoder/tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)

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
    data[segment_name]['encoding'] = [tokenizer(t, truncation=True, padding=True) for t in data[segment_name]['text']]

    pk.dump(data[segment_name], open(f"distilbert_dataset_{lang}_{segment_name}.pk", "wb"))
