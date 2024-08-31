# Investigating the Potential of Identifying Kidney Disease-Related Articles Using Transformer Models and Large Language Models

## Description

This repository contains the code for an MSc dissertation (COMP6200) at the University of Southampton, titled "Investigating the Potential of Identifying Kidney Disease-Related Articles Using Transformer Models and Large Language Models." The project is supervised by Dr. Mercedes Arguello Casteleiro.

This repository is structured as follows:

- **dataset_creation**: Contains the code for extracting the details of PubMed articles using PMIDs.
- **transformers**: Contains the code for hyperparameter tuning and fine-tuning the transformer models.
- **medprompt**: Contains the code for extracting the vector representations of the articles and the code for finding the most similar articles for dynamic few-shot prompting.
- **prompt_engineering**: Contains the code for generating LLMs' predictions for the articles and self-generated reasoning processes. Additionally, the prompt templates are stored in this directory.

## Installation

```
conda create -n my_env python=3.9
conda activate my_env
conda install pytorch=2.3.1 torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```

## Dataset Creation

Before running the code, users need to export the PMIDs of the articles from [PubMed](https://pubmed.ncbi.nlm.nih.gov/) (Select 'PMID' format). The code can be run using the following command:

```
python fetch_pubmed_articles.py --file_path <path_to_saved_pmids> --output_path <output_path>
```

## Transformers

For GridSearchCV, 5-fold cross-validation is used to find the best hyperparameters. The hyperparameter search is limited to learning rate and the number of epochs. The code can be run using the following command:

```
python gridsearchcv.py --dataset_path <path_to_dataset> --model <model_name> --batch_size <batch_size> --max_length <max_sequence_length>
```

As for fine-tuning the transformer models, the code can be run using the following command:

```
python fine_tuning.py --dataset_path <path_to_dataset> --model <model_name> --batch_size <batch_size> --max_length <max_sequence_length> --lr <learning_rate> --num_epochs <number_of_epochs> --seed <random_seed>
```

## MedPrompt

Before running the dynamic few-shot prompting code, users need to extract the vector representations of the articles, which will be saved in SQLite. The code can be run using the following command:

```
python extract_embedding.py --dataset_path <path_to_dataset> --model <model_name> --batch_size <batch_size> --max_length <max_sequence_length>
```

For finding the most similar articles for dynamic few-shot prompting, the code can be run using the following command:

```
python dynamic_few_shot_prompting.py --dataset_path <path_to_dataset> --database_path <path_to_database> --output_path <output_path>
```

## Prompt Engineering

`prompt_templates.json` contains the prompt templates for traditional, Chain-of-Thought (CoT), and Clue and Reasoning Prompting (CLUE) in both few-shot and zero-shot settings. Additionally, the prompts used for summarizing the PubMed articles and self-generated reasoning processes are stored in this file. The prompts are available for both classifying articles related to kidney disease and NCBI Disease Corpus sequence classification tasks.

The code for generating LLMs' predictions for the articles and self-generated reasoning processes can be run using the following command:

```
python prompting_llms.py --dataset_path <path_to_dataset> --prompt_temp <path_to_prompt_templates> --model <model_name> --prompt_type <name_of_prompt_in_prompt_templates> --output_path <output_path> --num_shots <number_of_few_shot_examples> --generated_examples_path <path_to_generated_examples> [Optional] --top_32_similar_articles_path <path_to_results_from_dynamic_few_shot_prompting> [Optional]
```
