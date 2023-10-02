import json
import random
from transformers import BertTokenizer, DataCollatorForLanguageModeling, TextDatasetForNextSentencePrediction
import random
import torch

# Load config file
with open("config.json", "r") as f:
        args = json.load(f)

# Load your JSON data
with open(args["data-file"], "r") as f:
    json_data = json.load(f)

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained(args["base-model"])

def create_nsp_dataset(json_data):
    sentences = [sentence["sentence"] for article in json_data["data"] for sentence in article["sentences"]]
    sentence_a = []
    sentence_b = []
    label = []
    # Create NSP dataset
    for i, _ in enumerate(sentences):
        start = random.randint(0, (len(sentences)-2))
        # Positive NSP example
        if random.random() >= 0.5:
            sentence_a.append(sentences[start])
            sentence_b.append(sentences[start+1])
            label.append(0)
        else:
    # Negative NSP example
            randint = random.randint(0,len(sentences)-1)
            if start != i:
                random_sentence = sentences[randint]
            else:
                random_sentence = sentences[randint-2]
            sentence_a.append(sentences[start])
            sentence_b.append(random_sentence)
            label.append(1)
    keys = list(range(len(sentence_a)))
#    dic = {k : {"sentence1": sent_a, "sentence2": sent_b, "label": lab} for k, sent_a, sent_b, lab in zip(keys, sentence_a, sentence_b, label)}
    dic = {"keys": keys, "Sentence1": sentence_a, "Sentence2": sentence_b, "labels": label}
    dic = dict(dic)
    return dic
