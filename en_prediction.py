import pickle as pk

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

checkpoint = "./results/checkpoint-37500"
model_name = 'distilbert-base-uncased'
data_dir = "./data/"
device_name = "cuda" if torch.cuda.is_available() else "cpu"
num_labels = 2

print("init model...")
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=num_labels).to(device_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


predictions = dict()
for segment_name in ['test', 'dev']:
    print(segment_name)
    data = pk.load(open(f"distilbert_dataset_en_{segment_name}.pk", "rb"))

    print("making predictions...")
    model.eval()  # set to evaluation mode
    predictions[segment_name] = list()
    with torch.no_grad():
        for batched_inputs in batch(data['text'], 20):
            encoding = tokenizer(batched_inputs, truncation=True, padding=True, return_tensors='pt').to(device_name)
            output = model(**encoding)
            _, pred = torch.max(output[0], dim=1)
            predictions[segment_name] += pred

torch.save(predictions, f"distilbert_predictions_en_binary.pt")
