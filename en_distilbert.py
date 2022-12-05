# Basic Python modules
from collections import defaultdict
import random
import pickle
import os
from collections import Counter


# For data manipulation and analysis
import pandas as pd
import numpy as np

# For machine learning tools and evaluation
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.model_selection import train_test_split

# For deep learning
# https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
import torch

# Transformer library
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments
# Transformer library
import json

model_output_path = ''
# Choose the model that we want to use (make sure to keep the cased/uncased consistent)
model_name = 'distilbert-base-uncased'

# Choose the GPU we want to process this script
device_name = 'cuda'

# This is the maximum number of tokens in any document sent to BERT
max_length = 512

print("reading files")
data_dir = "./data/"
train = [json.loads(line)
        for line in open(data_dir + 'dataset_en_train.json', 'r', encoding='utf-8')]
test = [json.loads(line)
        for line in open(data_dir + 'dataset_en_test.json', 'r', encoding='utf-8')]

train_texts = [r['review_body'] for r in train]
train_labels = [0 if int(r['stars'])<=3 else 1 for r in train]
print("train size: ", len(train_labels))

test_texts = [r['review_body'] for r in test]
test_labels = [0 if int(r['stars'])<=3 else 1 for r in test]
print("test size: ", len(test_labels))

print("load tokenizer")
# load the encoder/tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)

print("encoding texts")
# Pass training/testing sentences to tokenizer, truncate them if over max length, and add padding (PAD tokens up to 512)
train_encodings = tokenizer(train_texts,  truncation=True, padding=True)
test_encodings = tokenizer(test_texts,  truncation=True, padding=True)

print("making dataset structure")
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

train_dataset = SCDataset(train_encodings, train_labels)
test_dataset = SCDataset(test_encodings, test_labels)

print("init model")

model = DistilBertForSequenceClassification.from_pretrained(model_name).to(device_name)

# Define a custom evaluation function (this could be changes to return accuracy metrics)
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {
      'accuracy': acc,
    }

print("Set up trainer")
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=20,   # batch size for evaluation
    learning_rate=5e-5,              # initial learning rate for Adam optimizer
    warmup_steps=50,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
    evaluation_strategy='steps',
)
trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=test_dataset,            # evaluation dataset
    compute_metrics=compute_metrics      # custom evaluation function
)

print("training")

print("Set up training arguments")
trainer.train()