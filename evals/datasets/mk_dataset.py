# coding: utf-8
import argparse
import json
from sentence_transformers import (
    SentenceTransformer,
    evaluation,
)

parser = argparse.ArgumentParser(description="Used to evaluate a model on the dataset")
parser.add_argument(
    "--model",
    required=True,
    help="Indicate which encoder model you want to evaluate",
)
args = parser.parse_args()


def create_queries_dict(content):
    """
    Create a dictionary with 'id' as the key and 'query' as the value.

    Args:
    - content (list): List of dictionaries loaded from the JSONL file.

    Returns:
    - dict: Dictionary with 'id' as key and 'query' as value.
    """
    return {str(entry["id"]): str(entry["data"]["query"]) for entry in content}


def create_corpus_dict(content):
    """
    Create a dictionary with 'id' as the key and 'text' as the value.

    Args:
    - content (list): List of dictionaries loaded from the JSONL file.

    Returns:
    - dict: Dictionary with 'id' as key and 'text' as value.
    """
    return {str(entry["id"]): str(entry["data"]["text"]) for entry in content}


def create_relevant_docs_dict(content):
    """
    Create a dictionary with 'id' as the key and 'element_id' as the value.

    Args:
    - content (list): List of dictionaries loaded from the JSONL file.

    Returns:
    - dict: Dictionary with 'id' as key and 'element_id' as value.
    """
    return {str(entry["id"]): str(entry["id"]) for entry in content}


def load_jsonl(filename):
    """Loads JSON Lines file"""
    data = []
    with open(filename, "r", encoding="utf-8") as file:
        for line in file:
            data.append(json.loads(line))
    return data


lines = load_jsonl("processed_qa_pairs.jsonl")
model = SentenceTransformer(args.model)

queries = create_queries_dict(lines)
corpus = create_corpus_dict(lines)
relevant_docs = create_relevant_docs_dict(lines)

queries = {str(k): str(v) for k, v in queries.items()}
corpus = {str(k): str(v) for k, v in corpus.items()}
relevant_docs = {str(k): {str(v)} for k, v in relevant_docs.items()}

evaluator = evaluation.InformationRetrievalEvaluator(
    queries=queries,
    corpus=corpus,
    relevant_docs=relevant_docs,
    show_progress_bar=True,
    write_csv=True,
    main_score_function="cos_sim",
)
model.evaluate(evaluator, output_path="evaluation")
