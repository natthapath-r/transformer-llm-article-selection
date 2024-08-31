import os
import ast
import json
import time
import torch
import evaluate
import argparse
import transformers
from tqdm import tqdm
from datasets import load_from_disk, DatasetDict


def load_test_set(file_path: str) -> DatasetDict:
    dataset = load_from_disk(file_path)
    return dataset["test"]


def load_prompt_templates(file_path: str) -> dict:
    with open(file_path, "r") as f:
        prompt_templates = json.load(f)
    return prompt_templates


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="Path to the dataset",
    )
    parser.add_argument(
        "--prompt_temp",
        type=str,
        help="Path to the prompt templates",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model for prompting",
    )
    parser.add_argument("--prompt_type", type=str, help="The type of prompt to use")
    parser.add_argument("--output_path", type=str, help="Path to save the prompting results")
    parser.add_argument("--num_shots", type=int, help="Number of few-shot examples", default=0)
    parser.add_argument(
        "--generated_examples_path",
        type=str,
        help="Path to the generated examples for dynamic few-shot prompting",
        required=False,
    )
    parser.add_argument(
        "--top_32_similar_articles_path", type=str, help="Path to the top 32 most similar articles", required=False
    )
    args = parser.parse_args()

    # If num_shots is more than 0, then the generated_examples_path and top_32_similar_articles_path are required
    if args.num_shots > 0:
        if args.generated_examples_path is None or args.top_32_similar_articles_path is None:
            raise ValueError("Please provide the path to the generated examples and top 32 similar articles")

    logs_path = f"{args.output_path}/{args.model}_{args.prompt_type}_{args.num_shots}.json"

    # Get the test set
    test_set = load_test_set(args.dataset_path)
    # Get the prompt templates
    prompt_templates = load_prompt_templates(args.prompt_temp)
    prompt_messages = prompt_templates[args.prompt_type]

    # Load the generated examples for dynamic few-shot prompting
    with open(args.generated_examples_path, "r") as f:
        generated_examples_data = json.load(f)

    # Load top 32 most similar articles
    with open(args.top_32_similar_articles_path, "r") as f:
        top_32_most_similar_articles = json.load(f)

    # Get the details of articles in the test set
    pmid_list = test_set["PMID"]
    title_list = test_set["ArticleTitle"]
    abstract_list = test_set["Abstract"]
    gold_label_list = test_set["labels"]

    # Initialize the text generation pipeline
    pipeline = transformers.pipeline(
        "text-generation",
        model=args.model,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )
    terminators = [pipeline.tokenizer.eos_token_id, pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    # Remove None from terminators
    terminators = [x for x in terminators if x is not None]

    for index in tqdm(range(len(pmid_list))):
        # Initialize the prompting result logs
        if os.path.exists(logs_path):
            with open(logs_path, "r") as file:
                logs = json.load(file)
        else:
            logs = {}

        result_dict = {}
        pmid = pmid_list[index]
        if pmid not in logs:
            title = title_list[index]
            abstract = abstract_list[index]
            gold_label = gold_label_list[index]
            result_dict["Title"] = title
            result_dict["Abstract"] = abstract
            result_dict["GoldLabel"] = gold_label

            # Construct the prompt
            if args.num_shots == 0:
                prompt_messages[-1]["content"] = f"Title: {title}\n\nAbstract: {abstract}"
                # Verify the number of prompt messages in zero-shot setting
                if len(prompt_messages) != 2:
                    raise ValueError("Prompt messages length is not correct")
            else:
                prompt_messages = prompt_messages[:1]
                top_32_similar_pmids = top_32_most_similar_articles[pmid]["top_32_similar_pmids"]
                # Get the top k similar PMIDs and order them in ascending order based on the similarity score
                top_32_similar_pmids = [x[0] for x in top_32_similar_pmids[: args.num_shots]][::-1]

                # Adding the few-shot examples in the prompt
                for med_pmid in top_32_similar_pmids:
                    example_title = generated_examples_data[med_pmid]["Title"]
                    example_abstract = generated_examples_data[med_pmid]["Abstract"]
                    example_gold_label = generated_examples_data[med_pmid]["GoldLabel"]
                    prompt_messages.append(
                        {"role": "user", "content": f"Title: {example_title}\n\nAbstract: {example_abstract}"}
                    )
                    # Construct the prompt based on the prompt type
                    if "vanilla" in args.prompt_type:
                        formatted_content = '{{"class": {}}}'.format(example_gold_label)
                    elif "cot" in args.prompt_type:
                        example_cot = generated_examples_data[med_pmid]["explanation"]
                        formatted_content = '{{"explanation": "{}", "class": {}}}'.format(
                            example_cot, example_gold_label
                        )
                    elif "carp" in args.prompt_type:
                        example_clues = generated_examples_data[med_pmid]["Clues"]
                        example_reasoning = generated_examples_data[med_pmid]["reasoning"]
                        formatted_content = '{{"clues": {}, "reasoning": "{}", "class": {}}}'.format(
                            example_clues, example_reasoning, example_gold_label
                        )
                    else:
                        raise ValueError("Invalid prompt type")
                    prompt_messages.append({"role": "system", "content": formatted_content})

                # Adding the target article in the prompt
                prompt_messages.append({"role": "user", "content": f"Title: {title}\n\nAbstract: {abstract}"})
                # Verify the number of prompt messages in few-shot setting
                if len(prompt_messages) != 2 + 2 * args.num_shots:
                    raise ValueError("Prompt messages length is not correct")

            # Prompt the model
            start_time = time.time()
            response = pipeline(
                prompt_messages,
                eos_token_id=terminators,
                do_sample=True,
                temperature=1,
                top_p=1,
                max_length=8192,
            )
            end_time = time.time()
            result_dict["ResponseTime"] = end_time - start_time

            # Parse the response
            try:
                answer = ast.literal_eval(response[0]["generated_text"][-1]["content"])
                if "explanation" in answer:
                    result_dict["explanation"] = answer["explanation"]
                if "clues" in answer:
                    result_dict["clues"] = answer["clues"]
                if "reasoning" in answer:
                    result_dict["reasoning"] = answer["reasoning"]
                if "class" in answer:
                    result_dict["PredictedLabel"] = answer["class"]
                if "summary" in answer:
                    result_dict["summary"] = answer["summary"]
            except:
                result_dict["RawResponse"] = response[0]["generated_text"][-1]["content"]

            # Save the response
            logs[pmid] = result_dict
            with open(logs_path, "w") as f:
                json.dump(logs, f, indent=4)

    # Evaluate the model
    preds = []
    y = []

    with open(logs_path, "r") as file:
        logs = json.load(file)

    for pmid, result in logs.items():
        if pmid != "metrics" and "PredictedLabel" in result:
            preds.append(result["PredictedLabel"])
            y.append(result["GoldLabel"])

    f1_metric = evaluate.load("f1", average="macro")
    precision_metric = evaluate.load("precision", average="macro")
    recall_metric = evaluate.load("recall", average="macro")
    confusion_matrix_metric = evaluate.load("confusion_matrix")

    f1_metric.add_batch(predictions=preds, references=y)
    precision_metric.add_batch(predictions=preds, references=y)
    recall_metric.add_batch(predictions=preds, references=y)
    confusion_matrix_metric.add_batch(predictions=preds, references=y)

    score_dict = {
        "f1": f1_metric.compute()["f1"],
        "precision": precision_metric.compute()["precision"],
        "recall": recall_metric.compute()["recall"],
        "confusion_matrix": confusion_matrix_metric.compute()["confusion_matrix"].tolist(),
    }

    logs["metrics"] = score_dict

    with open(logs_path, "w") as f:
        json.dump(logs, f, indent=4)
