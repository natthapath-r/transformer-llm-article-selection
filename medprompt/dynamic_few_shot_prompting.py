import json
import sqlite3
import base64
import argparse
import numpy as np
from tqdm import tqdm
from datasets import load_from_disk
from sklearn.metrics.pairwise import cosine_similarity


def load_datasets(file_path: str):
    dataset = load_from_disk(file_path)
    return dataset


def decode_embeddings(encoded_embedding):
    return np.frombuffer(base64.b64decode(encoded_embedding), dtype=np.float32)


def compute_cosine_similarity(embeddings1, embeddings2):
    return cosine_similarity(embeddings1, embeddings2).diagonal()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="Path to the dataset",
    )
    parser.add_argument("--database_path", type=str, help="Path to the SQLite database")
    parser.add_argument("--output_path", type=str, help="Path to save the results")
    args = parser.parse_args()

    dataset = load_datasets(args.dataset_path)
    train_set = dataset["train"]
    test_set = dataset["test"]
    train_pmids = train_set["PMID"]
    test_pmids = test_set["PMID"]

    print(f"Number of train set PMIDs: {len(train_pmids)}")

    # Initialize SQLite connection
    conn = sqlite3.connect(args.database_path)
    cursor = conn.cursor()

    mem = {}

    for test_pmid in tqdm(test_pmids):
        # Get the embeddings and the label for the test PMID
        cursor.execute(
            """
            SELECT embedding, label FROM embeddings WHERE pmid = ?
        """,
            (test_pmid,),
        )
        test_row = cursor.fetchone()
        test_embedding = decode_embeddings(test_row[0])
        test_label = test_row[1]

        similarities = []
        for train_pmid in train_pmids:
            # Get the embeddings and the label for the train PMID
            cursor.execute(
                """
                SELECT embedding, label FROM embeddings WHERE pmid = ?
            """,
                (train_pmid,),
            )
            train_row = cursor.fetchone()
            train_embedding = decode_embeddings(train_row[0])
            train_label = train_row[1]

            if (test_pmid, train_pmid) in mem:
                similarity = mem[(test_pmid, train_pmid)]
            else:
                similarity = compute_cosine_similarity(test_embedding.reshape(1, -1), train_embedding.reshape(1, -1))[0]
                mem[(test_pmid, train_pmid)] = similarity
            similarities.append((train_pmid, similarity, train_label))

        # Find top 32 similar PMIDs
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_32_similar_pmids = similarities[:32]

        # Read the JSON file
        with open(args.output_path, "r") as f:
            logs = json.load(f)

        logs[test_pmid] = {
            "label": test_label,
            "top_32_similar_pmids": [(article[0], str(article[1])) for article in top_32_similar_pmids],
        }

        # Write the updated JSON file
        with open(args.output_path, "w") as f:
            json.dump(logs, f, indent=4)

    conn.close()
