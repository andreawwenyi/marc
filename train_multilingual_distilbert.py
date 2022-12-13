import argparse
import pickle as pk

import torch
# For machine learning tools and evaluation
from sklearn.metrics import accuracy_score
# Transformer library
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments

model_name = 'distilbert-base-multilingual-cased'
device_name = 'cuda'
data_dir = "./data/"


def finetune_mdistilbert(finetune_lang):
    model_output_path = f'./models/mdistilbert-{finetune_lang}'

    print("reading files")
    data = {
        "train": pk.load(open(f"clean_{finetune_lang}_train.pk", "rb")),
        "dev": pk.load(open(f"clean_{finetune_lang}_dev.pk", "rb")),
    }
    print("load tokenizer")
    # load the encoder/tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("making dataset...")

    class SCDataset(torch.utils.data.Dataset):
        def __init__(self, input_texts, labels):
            self.encodings = tokenizer(input_texts, truncation=True, padding=True, return_tensor='pt')
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: val[idx] for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    train_dataset = SCDataset(data['train']['text'], data['train']['binary_labels'])
    dev_dataset = SCDataset(data['dev']['text'], data['dev']['binary_labels'])

    print("init model")

    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device_name)

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
        output_dir='./results',  # output directory
        num_train_epochs=3,  # total number of training epochs
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=20,  # batch size for evaluation
        learning_rate=5e-5,  # initial learning rate for Adam optimizer
        warmup_steps=50,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir='./logs',  # directory for storing logs
        logging_steps=10,
        evaluation_strategy='steps',
    )
    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=dev_dataset,  # evaluation dataset
        compute_metrics=compute_metrics  # custom evaluation function
    )

    print("training")

    trainer.train()

    # save model
    model.save_pretrained(model_output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-lang", "--finetune_lang", required=True, type=str)
    args = parser.parse_args()

    finetune_mdistilbert(args.finetune_lang)
