import argparse
import torch
from datasets import load_dataset, get_dataset_split_names
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Trainer,
    TrainingArguments, 
    get_linear_schedule_with_warmup, 
    default_data_collator
)
from peft import get_peft_model, LoraConfig, TaskType
import numpy as np
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from accelerate import Accelerator
from rich.progress import Progress

def get_model(model_name="google-t5/t5-large", rank=8, alpha=32):
    # Load model and tokenizer
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = T5Tokenizer.from_pretrained(model_name)

    # Apply LoRA    
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=rank,  # Rank
        lora_alpha=alpha,
        lora_dropout=0.1,
        target_modules=["q", "v"]
    )
    
    model = get_peft_model(model, lora_config)
    model = model.to(device)
    model.print_trainable_parameters()
    return model, tokenizer

def preprocess_function(examples, tokenizer, max_length=256):
    inputs = examples[text_column]
    targets = examples[label_column]
    model_inputs = tokenizer(inputs, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
    labels = tokenizer(targets, max_length=2, padding="max_length", truncation=True, return_tensors="pt")
    labels = labels["input_ids"]
    labels[labels == tokenizer.pad_token_id] = -100
    model_inputs["labels"] = labels
    return model_inputs

def get_dataloader(batch_size=4):
    try:
        # Try loading from Hugging Face's datasets
        dataset = load_dataset("glue", "sst2", trust_remote_code=True)
        
        if not dataset:
            raise ValueError("Failed to load SST-2 dataset")
            
        classes = dataset["train"].features["label"].names
        
        # Map numerical labels to text labels
        dataset = dataset.map(
            lambda x: {"text_label": [classes[label] for label in x["label"]]},
            batched=True,
            num_proc=1
        )
        
        # Get model and tokenizer
        _, tokenizer = get_model()
        
        # Preprocess data with error handling
        processed_datasets = dataset.map(
            lambda x: preprocess_function(x, tokenizer),
            batched=True,
            num_proc=1,
            remove_columns=dataset["train"].column_names,
            load_from_cache_file=False,
            desc="Running tokenizer on dataset",
        )

        # Split data into train and eval
        train_dataset = processed_datasets["train"]
        eval_dataset = processed_datasets["validation"]

        # Create dataloaders
        train_dataloader = DataLoader(
            train_dataset, 
            shuffle=True, 
            collate_fn=default_data_collator, 
            batch_size=batch_size, 
            pin_memory=True
        )
        eval_dataloader = DataLoader(
            eval_dataset, 
            collate_fn=default_data_collator, 
            batch_size=batch_size, 
            pin_memory=True
        )

        return train_dataloader, eval_dataloader
        
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        print("Trying alternative loading method...")
        try:
            # Alternative: Try loading directly from Hugging Face's hub
            dataset = load_dataset("glue", "sst2", use_auth_token=False)
            # Continue with the same processing as above
            # [Rest of the processing code remains the same]
        except Exception as e2:
            print(f"Alternative loading method also failed: {str(e2)}")
            raise

def train(model, train_dataloader, eval_dataloader, lr=2e-5, num_epochs=3):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(train_dataloader) * num_epochs)
    )

    accelerator = Accelerator()
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    model.train()
    with Progress() as progress:
        task = progress.add_task("[red]Training...", total=num_epochs)

        for epoch in range(num_epochs):
            epoch_task = progress.add_task(f"Epoch {epoch}", total=len(train_dataloader))
            model.train()
            losses = []
            
            for step, batch in enumerate(train_dataloader):
                try:
                    input_ids = batch["input_ids"]
                    attention_mask = batch["attention_mask"]
                    labels = batch["labels"]
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    losses.append(loss.item())
                    accelerator.backward(loss)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    progress.update(epoch_task, advance=1)
                except Exception as e:
                    print(f"Error in training step: {str(e)}")
                    continue
            
            test_task = progress.add_task(f"Evaluation Epoch {epoch}", total=len(eval_dataloader))
            model.eval()
            eval_losses = []
            
            with torch.no_grad():
                for batch in eval_dataloader:
                    try:
                        outputs = model(**batch)
                        eval_losses.append(outputs.loss.item())
                        progress.update(test_task, advance=1)
                    except Exception as e:
                        print(f"Error in evaluation step: {str(e)}")
                        continue

            avg_train_loss = sum(losses) / len(losses) if losses else float('inf')
            avg_eval_loss = sum(eval_losses) / len(eval_losses) if eval_losses else float('inf')
            
            progress.update(task, advance=1)
            progress.print(f"Epoch: {epoch} Train Loss: {avg_train_loss:.4f} Eval Loss: {avg_eval_loss:.4f}")

        try:
            model.save_pretrained("t5-large-lora-fine-tuning-sst2")
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            # Attempt to save in a different location
            model.save_pretrained("./backup_model_save")

if __name__ == "__main__":
    # Set device
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # Constants
    text_column = "sentence"
    label_column = "text_label"
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Fine-tune T5 large on Sentiment analysis with LoRA.")
    parser.add_argument("--alpha", type=int, default=32, help="LoRA Alpha parameter")
    parser.add_argument("--rank", type=int, default=8, help="LoRA Rank parameter")
    parser.add_argument("--model_name", type=str, default="google-t5/t5-large", help="Model name")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs")

    args = parser.parse_args()

    try:
        model, _ = get_model(model_name=args.model_name, rank=args.rank, alpha=args.alpha)
        train_dataloader, eval_dataloader = get_dataloader(batch_size=args.batch_size)
        train(model, train_dataloader, eval_dataloader, lr=args.learning_rate, num_epochs=args.num_epochs)
    except Exception as e:
        print(f"Fatal error: {str(e)}")