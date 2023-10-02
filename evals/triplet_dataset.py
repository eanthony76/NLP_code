import argparse
import json
from json.decoder import JSONDecodeError
import random

from funcs import (
    gather_files,
    download_jsonl,
    save,
)
import boto3
from botocore.exceptions import NoCredentialsError

# Initialize argparse
parser = argparse.ArgumentParser(
    description="Script based on whether or not data already exists"
)
parser.add_argument(
    "--data-exists",
    choices=["true", "false"],
    required=True,
    help="Indicate if data exists.",
)

# pylint: disable="duplicate-code"
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

# pylint: disable="consider-using-f-string"
def main():
    """Uses existing dataset to create triplet set of data which
    includes a candidate, a positive example, and a negative example"""
    triplets = {}
    num = 0
    print(
        """What type of dataset are you using
    to create your triplets (qa/sts)?"""
    )
    data_type = input()
    if args.data_exists == "true":
        pass
    else:
        print(
            """You need to create the dataset first. Run the create_dataset script with
        --qa or --sts as the tag to create your dataset"""
        )

    files = gather_files.dataset(s3, config)
    for file in files:
        try:
            if file["Key"].endswith(f"processed_{data_type}_pairs.jsonl"):
                data = download_jsonl(s3, file["Key"], config)
                dsize = len(data) - 1
            else:
                continue

            if data is not None:
                for dic in data:
                    pos_query = str((dic["data"]["query"]))
                    neg_query = data[random.randint(0, dsize)]["data"]["query"]
                    text = dic["data"]["text"]
                    if pos_query != neg_query:
                        triplets[num] = [pos_query, neg_query, text]
                    else:
                        triplets[num] = [
                            pos_query,
                            str(data[random.randint(0, len(data))]["data"]["query"]),
                            text,
                        ]
                    num += 1
            save.s3jsonl(
                s3,
                config["bucket_name"],
                "{}/triplet_pairs_{}.jsonl".format(config["dataset_bucket"], type),
                triplets,
            )
        except JSONDecodeError:
            pass


if __name__ == "__main__":
    main()
