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
from sklearn.model_selection import StratifiedKFold
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset, DistributedSampler
from transformers import set_seed, AutoModelForSequenceClassification, get_scheduler


def load_datasets(file_path: str) -> DatasetDict:
    dataset = load_from_disk(file_path)
    return dataset


def tokenize_function(model_name: str, examples: DatasetDict, max_length: int) -> DatasetDict:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer(examples["Text"], padding="max_length", truncation=True, max_length=max_length)


def process_dataset(dataset: DatasetDict) -> DatasetDict:
    dataset = dataset.remove_columns(
        ["DateCompleted", "ArticleTitle", "Abstract", "PublicationTypeList", "MeshHeadingList", "PMID", "Text"]
    )
    dataset.set_format("torch")
    return dataset


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
    # Set seed for reproducibility
    seed = 42
    set_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    if not torch.cuda.is_available():
        raise ValueError("GPU not available")

    dist.init_process_group("nccl")

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
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum length of the input sequence")
    parser.add_argument("--local-rank", type=int, default=0, help="Local rank for distributed training")
    args = parser.parse_args()

    # Initialize the device
    device = torch.device(f"cuda:{args.local_rank}")

    # Load the dataset
    dataset = load_datasets(args.dataset_path)

    # Tokenize the dataset
    tokenized_dataset = dataset.map(
        lambda examples: tokenize_function(args.model, examples, args.max_length),
        batched=True,
    )

    # Process the dataset
    processed_dataset = process_dataset(tokenized_dataset)
    training_data = processed_dataset["train"]

    # Stratified 5-Fold Cross Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    gridsearch_params = {
        "lr": [1e-5, 3e-5, 5e-5, 7e-5, 1e-4],
        "num_epochs": [3, 4, 5],
    }
    max_avg_f1_score = 0
    best_model_name = None
    best_model_metrics = None

    gridsearch_power_usage = []

    # Start the timer for the grid search
    gridsearch_start_time = time.time()

    for lr in gridsearch_params["lr"]:
        for num_epochs in gridsearch_params["num_epochs"]:
            model_name = f"{args.model}_{args.batch_size}_{lr}_{num_epochs}"
            f1_scores = []

            # Start the timer
            start_time = time.time()

            for fold, (train_idx, val_idx) in enumerate(skf.split(training_data, training_data["labels"])):
                train_subset = Subset(training_data, train_idx)
                val_subset = Subset(training_data, val_idx)

                # Get the dataloaders
                train_sampler = DistributedSampler(train_subset, shuffle=True, seed=seed)
                train_dataloader = DataLoader(train_subset, batch_size=args.batch_size, sampler=train_sampler)
                val_dataloader = DataLoader(val_subset, batch_size=args.batch_size)

                model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2).to(device)
                model = DDP(model, device_ids=[args.local_rank])
                optimizer = AdamW(model.parameters(), lr=lr)
                num_training_steps = num_epochs * len(train_dataloader)
                lr_scheduler = get_scheduler(
                    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
                )

                # Training loop
                progress_bar = tqdm(range(num_training_steps))

                power_usage_list = []

                model.train()
                for epoch in range(num_epochs):
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

                    # Validation phase
                    model.eval()
                    total_val_loss = 0
                    with torch.no_grad():
                        for batch in val_dataloader:
                            batch = {k: v.to(device) for k, v in batch.items()}
                            outputs = model(**batch)
                            val_loss = outputs.loss.mean()
                            total_val_loss += val_loss.item()
                            power_usage = get_total_power_usage()
                            power_usage_list.append(power_usage)

                    average_val_loss = total_val_loss / len(val_dataloader)
                    progress_bar.set_description(
                        f"Epoch {epoch + 1} - Training Loss: {total_loss / len(train_dataloader):.4f}, Validation Loss: {average_val_loss:.4f}, Avg Power Usage: {sum(power_usage_list) / len(power_usage_list):.2f} W"
                    )

                    model.train()  # Make sure to reset to training mode after validation

                    gridsearch_power_usage.append(sum(power_usage_list) / len(power_usage_list))

                # Evaluate the model
                f1_metric = evaluate.load("f1", average="macro", experiment_id=model_name)
                model.eval()

                for batch in val_dataloader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    with torch.no_grad():
                        outputs = model(**batch)

                    logits = outputs.logits
                    predictions = torch.argmax(logits, dim=-1)

                    f1_metric.add_batch(predictions=predictions, references=batch["labels"])

                f1_score = f1_metric.compute()["f1"]
                progress_bar.set_description(f"Fold {fold + 1} - F1 Score: {f1_score:.4f} - model: {model_name}")
                progress_bar.close()
                f1_scores.append(f1_score)

            # End the timer
            k_fold_total_time = time.time() - start_time
            average_f1_score = sum(f1_scores) / len(f1_scores)

            if average_f1_score > max_avg_f1_score:
                max_avg_f1_score = average_f1_score
                best_model_name = model_name
                best_model_metrics = {
                    "f1_score": max_avg_f1_score,
                    "batch_size": args.batch_size,
                    "lr": lr,
                    "num_epochs": num_epochs,
                    "avg_training_time": k_fold_total_time / len(f1_scores),
                }

    # End the grid search timer
    gridsearch_total_time = time.time() - gridsearch_start_time

    # Export in JSON format
    with open(f"{best_model_name}.json", "w") as f:
        json.dump(best_model_metrics, f)

    # Report the grid search time in hours
    print("Total Grid Search Time: ", gridsearch_total_time / 3600, "hours")

    # Report the average power usage during grid search
    print("Average Power Usage: ", sum(gridsearch_power_usage) / len(gridsearch_power_usage), "W")
