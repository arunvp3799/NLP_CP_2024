from transformers import T5Tokenizer, T5ForConditionalGeneration
from peft import get_peft_model, PrefixTuningConfig, TaskType
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import default_data_collator, get_linear_schedule_with_warmup
import deepspeed
from rich.progress import Progress
from rich.console import Console

device = "cuda"
model_name_or_path = "google/flan-t5-xl"
tokenizer_name_or_path = "google/flan-t5-xl"

batch_size = 4  # Reduced batch size for testing
text_column = "sentence"
label_column = "text_label"
max_length = 256

tokenizer = T5Tokenizer.from_pretrained(tokenizer_name_or_path)

def preprocess_function(examples):
    inputs = examples[text_column]
    targets = examples[label_column]
    model_inputs = tokenizer(inputs, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
    labels = tokenizer(targets, max_length=2, padding="max_length", truncation=True, return_tensors="pt")
    labels = labels["input_ids"]
    labels[labels == tokenizer.pad_token_id] = -100
    model_inputs["labels"] = labels
    return model_inputs

def get_model():
    peft_config = PrefixTuningConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, num_virtual_tokens=64)
    model = T5ForConditionalGeneration.from_pretrained(model_name_or_path)
    model = get_peft_model(model, peft_config)
    return model

def get_dataloader():
    dataset = load_dataset("glue", "sst2")
    classes = dataset["train"].features["label"].names

    dataset = dataset.map(
        lambda x: {"text_label": [classes[label] for label in x["label"]]},
        batched=True,
        num_proc=1
    )

    processed_datasets = dataset.map(
        preprocess_function,
        batched=True,
        num_proc=1,
        remove_columns=dataset["train"].column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on dataset",
    )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation"]

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True)

    return train_dataloader, eval_dataloader

def train(model, train_dataloader, eval_dataloader):
    lr = 1e-5
    num_epochs = 5
    num_training_steps = len(train_dataloader) * num_epochs

    # Define the optimizer and learning rate scheduler with warmup
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=num_training_steps,
    )

    console = Console()

    # DeepSpeed initialization
    ds_config = "ds_config.json"  # Path to the DeepSpeed config file
    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        config=ds_config,
        model_parameters=model.parameters()
    )

    with Progress() as progress:
        train_task = progress.add_task("[red]Training...", total=num_epochs)
        for epoch in range(num_epochs):
            epoch_task = progress.add_task(f"Epoch {epoch + 1}/{num_epochs}", total=len(train_dataloader))
            model.train()
            
            for step, batch in enumerate(train_dataloader):
                batch = {k: v.to(model.device) for k, v in batch.items()}  # Move batch to the correct device
                outputs = model(**batch)
                loss = outputs.loss
                model.backward(loss)   # DeepSpeed manages backward and optimizer step
                model.step()           # Executes DeepSpeed's step
                
                progress.update(epoch_task, advance=1)

            # Evaluation step
            model.eval()
            eval_loss = 0.0
            with torch.no_grad():
                for step, batch in enumerate(eval_dataloader):
                    batch = {k: v.to(model.device) for k, v in batch.items()}
                    outputs = model(**batch)
                    eval_loss += outputs.loss.item()

            avg_eval_loss = eval_loss / len(eval_dataloader)
            progress.update(train_task, completed=num_epochs)
            model.save_checkpoint("sst2-t5-xl-prefix-tuning", tag=f"epoch_{epoch}")

            console.log(f"Epoch: {epoch} -- Train Loss: {loss.item()} -- Eval Loss: {avg_eval_loss}")

if __name__ == "__main__":
    model = get_model()
    train_dataloader, eval_dataloader = get_dataloader()
    train(model, train_dataloader, eval_dataloader)
