import json
import time
import torch
import argparse
import evaluate
import subprocess
from tqdm.auto import tqdm
from torch.optim import AdamW
import torch.distributed as dist
from transformers import AutoTokenizer
from datasets import load_from_disk, DatasetDict
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import set_seed, AutoModelForSequenceClassification, get_scheduler


def load_datasets(file_path: str) -> DatasetDict:
    dataset = load_from_disk(file_path)
    return dataset


def tokenize_function(model_name: str, examples: DatasetDict, max_length: int) -> DatasetDict:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer(examples["Text"], padding="max_length", truncation=True, max_length=max_length)


def process_dataset(dataset: DatasetDict, batch_size: int, seed: int = 42) -> DataLoader:
    dataset = dataset.remove_columns(
        ["DateCompleted", "ArticleTitle", "Abstract", "PublicationTypeList", "MeshHeadingList", "PMID", "Text"]
    )
    dataset.set_format("torch")

    # Create dataloaders
    train_sampler = DistributedSampler(dataset["train"], shuffle=True, seed=seed)
    train_dataloader = DataLoader(dataset["train"], batch_size=batch_size, sampler=train_sampler)
    test_dataloader = DataLoader(dataset["test"], batch_size=batch_size)

    return train_dataloader, test_dataloader


def get_gpu_power_usage() -> float:
    try:
        # Execute nvidia-smi command to get the power usage
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        # Extract the power usage from the output
        power_usage = 0
        for line in result.stdout.strip().split("\n"):
            power_usage += float(line)
        return power_usage
    except Exception as e:
        print(f"Error in getting GPU power usage: {e}")
        return None


def get_total_power_usage() -> float:
    local_power_usage = get_gpu_power_usage()
    total_power_usage = torch.tensor(local_power_usage, device=device)
    dist.all_reduce(total_power_usage, op=dist.ReduceOp.SUM)
    return total_power_usage.item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="Path to the dataset",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model to finetune",
    )
    parser.add_argument("--max_length", type=int, default=512, help="Maximum length of the input sequence")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for finetuning")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs to finetune")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate for finetuning")
    parser.add_argument("--local-rank", type=int, default=0, help="Local rank for distributed training")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")
    args = parser.parse_args()

    # Set seed for reproducibility
    seed = args.seed
    set_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    if not torch.cuda.is_available():
        raise ValueError("GPU not available")

    dist.init_process_group("nccl")

    output_name = f"{args.seed}_{args.model}_{args.batch_size}_{args.lr}_{args.num_epochs}"

    # Initialize the device
    device = torch.device(f"cuda:{args.local_rank}")

    # Load the dataset
    dataset = load_datasets(args.dataset_path)

    # Tokenize the dataset
    tokenized_dataset = dataset.map(
        lambda examples: tokenize_function(args.model, examples, args.max_length),
        batched=True,
    )

    # Process the dataset and get dataloaders
    train_dataloader, test_dataloader = process_dataset(tokenized_dataset, args.batch_size, seed)

    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2).to(device)
    model = DDP(model, device_ids=[args.local_rank])
    optimizer = AdamW(model.parameters(), lr=args.lr)
    num_training_steps = args.num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    # Training loop
    progress_bar = tqdm(range(num_training_steps))

    power_usage_list = []

    # Start the timer
    start_time = time.time()

    model.train()
    for epoch in range(args.num_epochs):
        total_loss = 0
        progress_bar = tqdm(range(len(train_dataloader)), desc=f"Epoch {epoch + 1}", unit="batch")
        for batch in train_dataloader:
            optimizer.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss.mean()
            loss.backward()

            optimizer.step()
            lr_scheduler.step()

            total_loss += loss.item()
            power_usage = get_total_power_usage()

            power_usage_list.append(power_usage)
            progress_bar.set_description(
                f"Epoch {epoch + 1} - Training Loss: {total_loss / (progress_bar.n + 1):.4f} - Power Usage: {power_usage:.2f} W"
            )
            progress_bar.update(1)

        progress_bar.set_description(
            f"Epoch {epoch + 1} - Training Loss: {total_loss / len(train_dataloader):.4f}, Avg Power Usage: {sum(power_usage_list) / len(power_usage_list):.2f} W"
        )
    progress_bar.close()

    # End the timer
    end_time = time.time()
    total_time = end_time - start_time

    # Evaluate the model
    f1_metric = evaluate.load("f1", average="macro")
    precision_metric = evaluate.load("precision", average="macro")
    recall_metric = evaluate.load("recall", average="macro")
    confusion_matrix_metric = evaluate.load("confusion_matrix")
    model.eval()

    for batch in test_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)

        f1_metric.add_batch(predictions=predictions, references=batch["labels"])
        precision_metric.add_batch(predictions=predictions, references=batch["labels"])
        recall_metric.add_batch(predictions=predictions, references=batch["labels"])
        confusion_matrix_metric.add_batch(predictions=predictions, references=batch["labels"])

    f1_score = f1_metric.compute()
    precision = precision_metric.compute()
    recall = recall_metric.compute()
    confusion_matrix = confusion_matrix_metric.compute()

    # Loggin the results
    logging_metrics = {
        "Model": args.model,
        "Batch Size": args.batch_size,
        "Learning Rate": args.lr,
        "Epochs": args.num_epochs,
        "Training Time": total_time,
        "F1 Score": f1_score["f1"],
        "Precision": precision["precision"],
        "Recall": recall["recall"],
        "Confusion Matrix": confusion_matrix["confusion_matrix"].tolist(),
        "Average Power Usage": sum(power_usage_list) / len(power_usage_list),
    }

    # Export in JSON format
    with open(f"{output_name}.json", "w") as f:
        json.dump(logging_metrics, f)

    # Save the model
    model.module.save_pretrained(output_name)
