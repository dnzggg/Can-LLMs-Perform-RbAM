import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

from sklearn.metrics import classification_report
from datasets import load_from_disk
from transformers import (
    RobertaTokenizerFast,
    RobertaForSequenceClassification,
    TrainingArguments,
    Trainer,
    set_seed
)
import numpy as np


set_seed(42)

model_id = "roberta-large"
name = "MArg"
dataset_loc = f"/vol/bitbucket/dg1822/CanLLMsPerformRbAM/Datasets/{name}Dataset"

tokenizer = RobertaTokenizerFast.from_pretrained(model_id, cache_dir="cache/")
model = RobertaForSequenceClassification.from_pretrained(model_id, cache_dir="cache/", num_labels=3)

dataset_plain = load_from_disk(dataset_loc)
dataset_plain = dataset_plain.map(lambda data: {"text": f"Arg1:{data['arg1']}, Arg2: {data['arg2']}"})
dataset_plain = dataset_plain.rename_column("support", "label")
dataset = dataset_plain.train_test_split(seed=42, test_size=0.25)

train_dataset = dataset['train']
val_dataset = dataset["test"]


def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)


train_dataset = train_dataset.map(tokenize, batched=True, batch_size=len(train_dataset))
val_dataset = val_dataset.map(tokenize, batched=True, batch_size=len(val_dataset))

train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])


repository_id = f"/data/dg1822/Roberta-{name}"
training_args = TrainingArguments(
    output_dir=repository_id,
    num_train_epochs=50,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    evaluation_strategy="epoch",
    logging_dir=f"{repository_id}/logs",
    logging_strategy="steps",
    logging_steps=10,
    learning_rate=1e-5,
    warmup_steps=500,
    save_strategy="epoch",
    load_best_model_at_end=True,
    save_total_limit=2
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

trainer.train()
