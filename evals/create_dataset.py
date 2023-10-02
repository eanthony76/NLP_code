import json
from json import JSONDecodeError
import argparse

import boto3
from transformers import pipeline
from botocore.exceptions import NoCredentialsError
from funcs import (
    Prompt,
    save,
    get_file_obj,
    gather_files,
    clean_generated_text,
)

# Initialize argparse
parser = argparse.ArgumentParser(description="Script used to create pairs")
parser.add_argument(
    "--type",
    choices=["sts", "qa"],
    required=True,
    help="Indicate which dataset you want to create",
)

args = parser.parse_args()
# Read the config file
with open("config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

# Initialize the S3 client
print("Checking for AWS Credentials")
try:
    s3 = boto3.client("s3")
except NoCredentialsError:
    print("Client Initialization Failed - Check your AWS Credentials")

# Initialize the pipeline
model_pipeline = pipeline(config["pipeline_task"], model=config["model_name"], device=0)


def main():
    """Streams files from S3, processes, and puts through a pipeline to create
    query-response pairs"""
    # List the files in the S3 bucket
    files = gather_files.raw(s3_client=s3, config=config)
    parent_dict = {}
    key_counter = 0
    for file in files:
        if file["Key"].endswith(".zip"):
            pass
        # Download and process each file
        file_obj = get_file_obj(s3_client=s3, file=file, config=config)
        try:
            sentences = json.loads(file_obj["Body"].read())
        except JSONDecodeError:
            sentences = None
        except UnicodeDecodeError:
            sentences = None
        # Run the pipeline to create queries to go with responses
        if sentences is not None:
            for sent in sentences:
                if args.type == "sts":
                    query = model_pipeline(
                        Prompt.query_prompt(
                            config["sts_prompt"], sent[config["sentences_field"]]
                        )
                    )
                elif args.type == "qa":
                    query = model_pipeline(
                        Prompt.query_prompt(
                            config["query_prompt"], sent[config["sentences_field"]]
                        )
                    )
                query = clean_generated_text.T5(query)
                sent["query"] = query
                if args.type == "qa":
                    response = model_pipeline(
                        Prompt.response_prompt(
                            config["answer_prompt"],
                            sent[config["sentences_field"]],
                            query,
                        )
                    )
                    response = clean_generated_text.T5(response)
                    sent["response"] = response
                parent_dict[key_counter] = sent
                key_counter += 1
    save.s3jsonl(
        s3,
        config["bucket_name"],
        f'{config["dataset_bucket"]}/{args.type}_pairs.jsonl',
        parent_dict,
    )


if __name__ == "__main__":
    main()
