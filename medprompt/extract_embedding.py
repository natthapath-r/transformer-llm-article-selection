import torch
import base64
import sqlite3
import argparse
from datasets import load_from_disk, DatasetDict
from transformers import set_seed, AutoTokenizer, AutoModel
from torch.utils.data import DataLoader


def load_datasets(file_path: str) -> DatasetDict:
    dataset = load_from_disk(file_path)
    return dataset


def tokenize_function(tokenizer, examples, max_length: int):
    return tokenizer(examples["Text"], padding="max_length", truncation=True, max_length=max_length)


def process_dataset(dataset: DatasetDict, batch_size: int = 16):
    # Remove unnecessary fields
    dataset = dataset.remove_columns(
        ["DateCompleted", "ArticleTitle", "Abstract", "PublicationTypeList", "MeshHeadingList", "Text"]
    )
    dataset.set_format("torch")

    # Create dataloaders
    train_dataloader = DataLoader(dataset["train"], batch_size=batch_size)
    test_dataloader = DataLoader(dataset["test"], batch_size=batch_size)

    return train_dataloader, test_dataloader


# Function to get embeddings for an article
def get_embeddings(batch, model, device):
    # Move the inputs to the GPU
    inputs = {key: value.to(device) for key, value in batch.items() if key not in ["PMID", "labels"]}

    # Get the token embeddings
    with torch.no_grad():
        outputs = model(**inputs)

    # The last hidden state contains the word embeddings
    embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    return embeddings


# Function to encode embeddings to store in the database
def encode_embeddings(embeddings):
    return base64.b64encode(embeddings).decode("utf-8")


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, help="Path to the dataset")
    parser.add_argument(
        "--model",
        type=str,
        help="Model to create embeddings with",
    )
    parser.add_argument("--max_length", type=int, default=512, help="Maximum length of the input sequence")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for dataloader")
    args = parser.parse_args()

    # Set seed for reproducibility
    seed = 42
    set_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    # Initialize the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the dataset
    dataset = load_datasets(args.dataset_path)

    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device)

    # Tokenize the dataset
    tokenized_dataset = dataset.map(
        lambda examples: tokenize_function(tokenizer, examples, args.max_length),
        batched=True,
    )

    # Process the dataset and get dataloaders
    train_dataloader, test_dataloader = process_dataset(tokenized_dataset, args.batch_size)

    # Initialize SQLite database
    conn = sqlite3.connect("embedding.db")
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS embeddings (
            id TEXT PRIMARY KEY,
            embedding TEXT,
            label INTEGER
        )
    """
    )
    conn.commit()

    # Apply the embedding extraction across the dataset
    for dataloader in [train_dataloader, test_dataloader]:
        for batch in dataloader:
            embeddings = get_embeddings(batch, model, device)
            for idx, pmid in enumerate(batch["PMID"]):
                encoded_embedding = encode_embeddings(embeddings[idx])
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO embeddings (id, embedding, label)
                    VALUES (?, ?, ?)
                """,
                    (pmid, encoded_embedding, int(batch["labels"][idx].item())),
                )

    conn.commit()
    conn.close()
