import json
import pickle as pk

import torch
from transformers import DistilBertTokenizerFast

data_dir = "./data/"
model_name = 'distilbert-base-uncased'
lang = 'en'


class SCDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


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
    data[segment_name]['encoding'] = tokenizer(data[segment_name]['text'], truncation=True, padding=True)

    print("generating dataset structure...")
    data[segment_name]['torch_dataset'] = SCDataset(data[segment_name]['encoding'], data[segment_name]['labels'])
    pk.dump(data[segment_name], open(f"distilbert_dataset_{lang}_{segment_name}.pk", "wb"))
