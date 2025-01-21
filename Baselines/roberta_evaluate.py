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
    AutoConfig,
    set_seed
)
import numpy as np
import glob

from datasets import load_from_disk, Dataset


set_seed(42)
model_id = "roberta-large"
# name = "Microtexts"
tokenizer = RobertaTokenizerFast.from_pretrained(model_id, cache_dir="cache/")
# model = RobertaForSequenceClassification.from_pretrained(f"/data/dg1822/Roberta-{name}/checkpoint-336", cache_dir="cache/")

for name in ["Web"]:
# for name in ["NR_NixonKennedy", "NR_UKP", "NR_Web", "NR_MArg"]:
# for name in ["Essays", "NixonKennedy", "CDCP", "UKP", "DebatepediaProcon", "IBMDebater", "ComArg", "Microtexts", "Kialo", "Web"]:
    file = f"/data/dg1822/Roberta-{name}/checkpoint-best"
    model = RobertaForSequenceClassification.from_pretrained(file, cache_dir="cache/")
  
    for name2 in ["MArg"]:
    # for name2 in ["NR_NixonKennedy", "NR_UKP", "NR_Web", "NR_MArg"]:
    # for name2 in ["EssaysRelation", "NixonKennedy", "CDCP", "UKP", "DebatepediaProcon", "IBMDebater", "ComArg", "Microtexts", "Kialo", "Web"]:
        if name == name2 or (name=="Essays" and name2=="EssaysRelation"):
            continue
        dataset_loc = f"/vol/bitbucket/dg1822/CanLLMsPerformRbAM/Datasets/{name2}Dataset"
        print(file, dataset_loc)
        dataset_plain = load_from_disk(dataset_loc)
        dataset_plain = dataset_plain.map(lambda data: {"text": f"Arg1:{data['arg1']}, Arg2: {data['arg2']}"})
        dataset_plain = dataset_plain.rename_column("support", "label")
        # dataset = dataset_plain.train_test_split(seed=42)
    
    
        def tokenize(batch):
            return tokenizer(batch["text"], padding=True, truncation=True)
    
    
        dataset_plain = dataset_plain.map(tokenize, batched=True, batch_size=len(dataset_plain))
        dataset_plain.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    
        config = AutoConfig.from_pretrained(model_id)
        
        repository_id = "Roberta"
        training_args = TrainingArguments(
            output_dir=repository_id,
            num_train_epochs=50,
            per_device_train_batch_size=16,
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
        )
    
        pred = trainer.predict(dataset_plain)
        dataset = load_from_disk(dataset_loc)
        result = np.argmax(pred.predictions, axis=-1)
        results = {"arg1": dataset["arg1"], "arg2": dataset["arg2"], "support": dataset["support"], "result": result}
        result_dataset = Dataset.from_dict(results)
        print(len(results["arg1"]), len(results["result"]))
        result_dataset.save_to_disk(f"Results/Result-Roberta-{name}-{name2}Dataset")
        print(classification_report(dataset_plain["label"], result, digits=4, zero_division=True))

