import pickle as pk

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

checkpoint = "./results/checkpoint-37500"
model_name = 'distilbert-base-uncased'
data_dir = "./data/"
device_name = 'cuda'
num_labels = 2
data = {}

print("init model...")
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=num_labels).to(device_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


predictions = dict()
for segment_name in ['test', 'dev']:
    print(segment_name)
    data = pk.load(open(f"distilbert_dataset_en_{segment_name}.pk", "rb"))

    print("encoding...")
    inputs = data['text']
    encoding = tokenizer(inputs, truncation=True, padding=True, return_tensors='pt')
    print("making predictions...")
    model.eval()  # set to evaluation mode
    with torch.no_grad():
        output = model(**encoding)
        _, pred = torch.max(output[0], dim=1)
    predictions[segment_name] = pred

pk.dump(predictions, open(f"distilbert_predictions_en_binary.pk", "wb"))
