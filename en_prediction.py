import json
import pickle as pk

import torch
import transformers
from sklearn.metrics import classification_report

checkpoint = "./results/checkpoint-37500"
model_name = 'distilbert-base-uncased'
data_dir = "./data/"
device_name = 'cuda'
data = {}


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


print("init model...")
model = transformers.AutoModel.from_pretrained(checkpoint).to(device_name)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

for dataset_name in ['test', 'dev']:
    print(dataset_name)
    raw_data = [json.loads(line) for line in open(data_dir + f'dataset_en_{dataset_name}.json', 'r', encoding='utf-8')]
    data[dataset_name] = {
        "text": [r['review_body'] for r in raw_data],
        "binary_labels": [0 if int(r['stars']) <= 3 else 1 for r in raw_data],
        "original_labels": [r['stars'] for r in raw_data],
    }
    print("generating encoding...")
    data[dataset_name]['encoding'] = [tokenizer(t, truncation=True, padding=True, return_tensors='pt') for t in
                                      data[dataset_name]['text']]
    # print("generating dataset structure...")
    # data[dataset_name]['torch_dataset'] = SCDataset(data[dataset_name]['encoding'], data[dataset_name]['binary_labels'])
    pk.dump(data[dataset_name], open(f"distilbert_dataset_en_{dataset_name}.pk", "wb"))

predictions = dict()
for dataset_name in ['test', 'dev']:
    print("making predictions...")
    predictions[dataset_name] = list()
    for enc in data[dataset_name]['encoding']:
        predictions[dataset_name].append(model(**enc))
pk.dump(predictions, open(f"distilbert_predictions_en.pk", "wb"))
    # actual_predicted_labels = predicted_labels.predictions.argmax(-1)
    # print(classification_report(predicted_labels.label_ids.flatten(), actual_predicted_labels.flatten()))
