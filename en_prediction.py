import json
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import torch
import transformers

checkpoint = "./results/checkpoint-37500"

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
model = transformers.AutoModel(checkpoint).to(device_name)
tokenizer = transformers.AutoTokenizer(checkpoint)

for dataset_name in ['test', 'dev']:
    print(dataset_name)
    raw_data = [json.loads(line) for line in open(data_dir + f'dataset_en_{dataset_name}.json', 'r', encoding='utf-8')]
    data[dataset_name] = {
        "text": [r['review_body'] for r in raw_data],
        "labels": [0 if int(r['stars']) <= 3 else 1 for r in raw_data]
    }
    print("generating encoding...")
    data[dataset_name]['encoding'] = tokenizer(data[dataset_name]['text'], truncation=True, padding=True)

    print("generating dataset structure...")
    data[dataset_name]['torch_dataset'] = SCDataset(data[dataset_name]['encoding'], data[dataset_name]['labels'])
    print("making predictions...")
    predicted_labels = model.predict(data[dataset_name]['torch_dataset'])
    actual_predicted_labels = predicted_labels.predictions.argmax(-1)
    print(classification_report(predicted_labels.label_ids.flatten(), actual_predicted_labels.flatten()))