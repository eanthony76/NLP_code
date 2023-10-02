import json
from json.decoder import JSONDecodeError
import tqdm

from funcs import (
    gather_files,
    create_corpus,
    download_jsonl,
    save,
    BM25,
)
import nltk
import boto3
from botocore.exceptions import NoCredentialsError

with open("config.json", "r", encoding="utf-8") as fIn:
    config = json.load(fIn)

try:
    s3 = boto3.client("s3")
except NoCredentialsError:
    print("Client Initialization Failed - Check your AWS Credentials")

# pylint: disable="consider-using-f-string"
def main():
    """Perform BM25 dropout which will remove rows from the
    dataset which do not appear in the top n (default n=2)
    results of a BM25 search"""
    files = gather_files.processed(s3, config)
    for file in files:
        try:
            if file["Key"].endswith(".jsonl"):
                data = download_jsonl(s3, file["Key"], config)
                corpus = create_corpus(data)
            else:
                data = None
            to_delete = []
            if data is not None:
                for idx, dic in tqdm.tqdm(enumerate(data)):
                    query = str((dic["data"]["query"]))
                    tokenized_query = nltk.word_tokenize(query)
                    if idx % 500 == 0:
                        print(f"Checking query {idx}")
                    if dic["data"]["text"] not in BM25.rank(
                        tokenized_query, corpus, top_n=2
                    ):
                        to_delete.append(dic)
                for deletion in to_delete:
                    data.remove(deletion)
                print(
                    f"Your data contain {len(data)} usable training examples after BM25 dropout"
                )
                save.processed_jsonl(
                    s3,
                    config["bucket_name"],
                    "{}/processed_qa_pairs.jsonl".format(config["dataset_bucket"]),
                    data,
                )
            else:
                print("No data found. Check your jsonl file(s) and try again")
        except JSONDecodeError:
            pass


if __name__ == "__main__":
    main()
