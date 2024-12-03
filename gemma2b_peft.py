import torch
from tqdm import tqdm
from accelerate import Accelerator
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW, get_linear_schedule_with_warmup
from peft import get_peft_model, PromptTuningConfig, PromptTuningInit, TaskType

global DEVICE 
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

hf_token = "HF_TOKEN_ID"
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b", token=hf_token)

def parse_dataset(dataset):

    prompt_template = """
    Classify the sentiment of the following sentence as Positive or Negative: 
    {sentence}
    Output: """

    label_map = {0: "Negative", 1: "Positive"}
    prompt_dataset = {}
    prompt_dataset["prompt"] = []
    prompt_dataset["label"] = []
    for sample in dataset:
        prompt = prompt_template.format(sentence=sample["sentence"])
        prompt_dataset["prompt"].append(prompt)
        prompt_dataset["label"].append(label_map[sample["label"]])
    
    return prompt_dataset

def preprocess_dataset(exmaple):
    prompt_max_length = 512
    target_max_length = 2
    inputs = tokenizer(exmaple["prompt"], return_tensors="pt", padding="max_length", truncation=True, max_length=prompt_max_length)
    targets = tokenizer(exmaple["label"], return_tensors="pt", padding="max_length", truncation=True, max_length=target_max_length)
    labels = targets["input_ids"]
    labels[labels == tokenizer.pad_token_id] = -100
    inputs["labels"] = labels
    return inputs

def prepare_data():
    
    dataset = load_dataset("amazon_polarity")
    train = Dataset.from_dict(parse_dataset(dataset["train"]))
    validation = Dataset.from_dict(parse_dataset(dataset["validation"]))

    train = train.map(preprocess_dataset, batched=True)
    validation = validation.map(preprocess_dataset, batched=True)

    train.set_format(type="torch")
    validation.set_format(type="torch")

    train_loader = DataLoader(train, batch_size=8, shuffle=True)
    validation_loader = DataLoader(validation, batch_size=8)

    return train_loader, validation_loader

def get_model():

    model = AutoModelForCausalLM.from_pretrained("google/gemma-2b", token=hf_token)
    peft_config = PromptTuningConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        prompt_tuning_init=PromptTuningInit.TEXT,
        num_virtual_tokens=100,
        prompt_tuning_init_text=PromptTuningInit.RANDOM,
        inference_mode=False,
        tokenizer_name_or_path="google/gemma-2b",
    )

    model = get_peft_model(model, peft_config)
    return model

def train():

    num_epochs = 3
    train_loader, validation_loader = prepare_data()
    model = get_model().to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    lr_scheduler = lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(train_loader) * num_epochs),
    )

    accelerator = Accelerator()
    model, optimizer, train_loader, lr_scheduler = accelerator.prepare(model, optimizer, train_loader, lr_scheduler)

    model.train()
    for epoch in range(num_epochs):
        for batch in tqdm(train_loader):
            inputs = {k: v.to(DEVICE) for k, v in batch.items() if k != "labels"}
            labels = batch["labels"].to(DEVICE)
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            ## Checkpoint
            if (train_loader.iterations % 100) == 0:
                print(f"Iteration {train_loader.iterations}, Loss: {loss.item()}")
                model.save_pretrained("sst2-checkpoint")
        print(f"Epoch {epoch} completed")

    model.save_pretrained("sst2-final")    

if __name__ == "__main__":
    train()