from io import BytesIO
import zipfile
import json

from botocore.exceptions import NoCredentialsError
from rank_bm25 import BM25Okapi
import nltk


class Prompt:
    """This class contains different prompts for Huggingface pipelines"""

    def __init__(self, prompt, text, text2=None):
        """Initialize a Prompt object."""
        self.prompt = prompt
        self.text = text
        self.text2 = text2

    @staticmethod
    def query_prompt(prompt, text):
        """Generate a query prompt based on the provided prompt and text."""
        return f"{prompt}: {text}"

    @staticmethod
    def response_prompt(prompt, text, text2):
        """Generate a response prompt based on the provided prompt,
        text, and additional context."""
        return f"{prompt}: question: {text}, context: {text2}"


# pylint: disable="consider-using-with"
def unzip_file(s3_client, config, file):
    """Unzip a file from an S3 bucket and upload its contents to
    a specified location."""
    zip_obj = s3_client.get_object(Bucket=config["bucket_name"], Key=file["Key"])
    buffer = BytesIO(zip_obj["Body"].read())
    zipp = zipfile.ZipFile(buffer)
    for filename in zipp.namelist():
        s3_client.upload_fileobj(
            zipp.open(filename),
            Bucket=config["output_bucket_name"],
            Key=f'{config["s3_input_prefix"]}/{filename}',
        )
        print("File Uploaded")


# pylint: disable="invalid-name"
class save:
    """Saves file types in locaton specified in config file"""

    @staticmethod
    def s3json(s3_client, bucket_name, file_name, data):
        """Save the provided data as JSON to an S3 bucket."""
        try:
            s3_client.put_object(
                Body=json.dumps(data), Bucket=bucket_name, Key=file_name
            )
        except NoCredentialsError:
            print("S3 Upload Failed - Check your AWS Credentials")

    @staticmethod
    def s3jsonl(s3_client, bucket_name, file_name, data):
        """Save the provided data as JSONL to an S3 bucket."""
        jsonl_buffer = BytesIO()
        for key, value in data.items():
            entry = {"id": key, "data": value}
            json_line = json.dumps(entry) + "\n"
            jsonl_buffer.write(json_line.encode("utf-8"))
        jsonl_buffer.seek(0)
        try:
            s3_client.upload_fileobj(jsonl_buffer, bucket_name, file_name)
            print("S3 Upload Successful")
        except NoCredentialsError:
            print("S3 Upload Failed - Check your AWS Credentials")

    @staticmethod
    def processed_jsonl(s3_client, bucket_name, file_name, data):
        """Upload processed data as JSONL to an S3 bucket."""
        jsonl_buffer = BytesIO()
        for dic in data:
            entry = dic
            json_line = json.dumps(entry) + "\n"
            jsonl_buffer.write(json_line.encode("utf-8"))
        jsonl_buffer.seek(0)
        try:
            s3_client.upload_fileobj(jsonl_buffer, bucket_name, file_name)
            print("S3 Upload Successful")
        except NoCredentialsError:
            print("S3 Upload Failed - Check your AWS Credentials")


def get_file_obj(s3_client, file, config):
    """Retrieve a file object from an S3 bucket."""
    return s3_client.get_object(Bucket=config["bucket_name"], Key=file["Key"])


class gather_files:
    """Gathers files from provided S3 bucket"""

    @staticmethod
    def raw(s3_client, config):
        """Gather raw files from an S3 bucket, unzip them if necessary, and return the list of
        files."""
        files = s3_client.list_objects(
            Bucket=config["bucket_name"], Prefix=config["s3_input_prefix"]
        )["Contents"]
        for file in files:
            if file["Key"].endswith(".zip"):
                unzip_file(s3_client, config, file)
        files = s3_client.list_objects(
            Bucket=config["bucket_name"], Prefix=config["s3_input_prefix"]
        )["Contents"]
        return files

    @staticmethod
    def processed(s3_client, config):
        """Gather processed files from an S3 bucket, unzip them if necessary, and return the
        list of files."""
        files = s3_client.list_objects(
            Bucket=config["bucket_name"], Prefix=config["processed_bucket"]
        )["Contents"]
        for file in files:
            if file["Key"].endswith(".zip"):
                unzip_file(s3_client, config, file)
        files = s3_client.list_objects(
            Bucket=config["bucket_name"], Prefix=config["processed_bucket"]
        )["Contents"]
        return files

    @staticmethod
    def dataset(s3_client, config):
        """Gather dataset files from an S3 bucket, unzip them if necessary, and return
        the list of files."""
        files = s3_client.list_objects(
            Bucket=config["bucket_name"], Prefix=config["dataset_bucket"]
        )["Contents"]
        for file in files:
            if file["Key"].endswith(".zip"):
                unzip_file(s3_client, config, file)
        files = s3_client.list_objects(
            Bucket=config["bucket_name"], Prefix=config["dataset_bucket"]
        )["Contents"]
        return files


class BM25:
    """Uses OkapiBM25 to perform a BM25 search for the purposes of filtering
    bad training examples out of the dataset"""

    @staticmethod
    def tokenize(corpus):
        """Tokenize the given corpus using NLTK and return a BM25 object."""
        tokenized_corpus = [nltk.word_tokenize(c) for c in corpus]
        return BM25Okapi(tokenized_corpus)

    @staticmethod
    def rank(query, corpus, top_n=5):
        """Rank the provided corpus based on the given query and return the top N results."""
        bm25_corpus = BM25.tokenize(corpus)
        return bm25_corpus.get_top_n(query, corpus, n=top_n)

    @staticmethod
    def print_rankings(rankings):
        """Print the provided rankings."""
        for r in rankings:
            print(r)


def download_jsonl(s3_client, key, config):
    """Download a JSONL file from an S3 bucket and return
    the data as a list of dictionaries."""
    jsonl_buffer = BytesIO()
    s3_client.download_fileobj(config["bucket_name"], key, jsonl_buffer)
    jsonl_buffer.seek(0)
    data = []
    for line in jsonl_buffer:
        data.append(json.loads(line.decode("utf-8")))
    return data


def create_corpus(data):
    """Create a corpus from the provided data."""
    return [dic["data"]["text"] for dic in data]


# pylint: disable="too-few-public-methods", "invalid-name"
class clean_generated_text:
    """Clean text from unwanted characters"""

    @staticmethod
    def T5(text):
        """Clean text generated by T5 model"""
        text = str(text).replace("[{'generated_text': '", "")
        text = text.replace("'}]", "")
        return text
