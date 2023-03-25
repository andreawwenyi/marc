import argparse
import pickle as pk

import torch
# For machine learning tools and evaluation
from sklearn.metrics import accuracy_score
# Transformer library
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments

model_name = 'xlm-roberta-base'
device_name = 'cuda'
data_dir = "./data/"


# Define a custom evaluation function (this could be changes to return accuracy metrics)
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
    }


def finetune(model_lang):
    model_output_path = f'./models/{model_name}/{model_lang}'

    print("reading files")
    original_dataset = {
        "train": pk.load(open(data_dir + f"clean_{model_lang}_train.pk", "rb")),
        "dev": pk.load(open(data_dir + f"clean_{model_lang}_dev.pk", "rb")),
    }
    print("load tokenizer")
    # load the encoder/tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("making dataset...")

    class SCDataset(torch.utils.data.Dataset):
        def __init__(self, input_texts, labels):
            self.encodings = tokenizer(input_texts, truncation=True, padding=True, return_tensors='pt')
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: val[idx] for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    lm_dataset = {"train": SCDataset(original_dataset['train']['text'], original_dataset['train']['binary_labels']),
                  "dev": SCDataset(original_dataset['dev']['text'], original_dataset['dev']['binary_labels'])}

    print("init model")

    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device_name)

    print("Set up trainer")
    training_args = TrainingArguments(
        output_dir=model_output_path,  # output directory
        num_train_epochs=10,
        evaluation_strategy="steps",
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_dir='./logs',  # directory for storing logs
        logging_steps=10,
        load_best_model_at_end=True
    )
    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=lm_dataset['train'],  # training dataset
        eval_dataset=lm_dataset['dev'],  # evaluation dataset
        compute_metrics=compute_metrics  # custom evaluation function
    )

    print("training")

    trainer.train()

    # save model
    model.save_pretrained(model_output_path + "/best")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-lang", "--model-lang", required=True, type=str)
    args = parser.parse_args()
    finetune(args.model_lang)
