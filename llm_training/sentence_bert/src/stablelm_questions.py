import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from prompting import prompt
from tqdm import tqdm
import json
import pandas as pd
import nltk
#nltk.download('punkt')  # Download the necessary NLTK data
from nltk.tokenize import sent_tokenize
import os


with open('config.json') as config_file:
        args = json.load(config_file)

def create_dataset(datadir, text_col=None):
        sentences = []
        if str(datadir).endswith('.csv'):
                df = pd.read_csv(datadir)
                sentences = df[args['datacol']]
        elif str(datadir).endswith('/'):
                files = os.listdir(datadir)
                for file in files:
                        if str(file).endswith('.json'):
                                data = os.path.join(datadir, file)
                                with open(data) as f:
                                        file = json.load(f)
                                        sentences.extend([sentence["sentence"] for article in file["data"] for sentence in article["sentences"]])
                        else:
                                pass

        else:
                print('data not in usable format. Current formats include .csv and .json')
        return sentences

data = create_dataset(args['datafile'])

tokenizer = AutoTokenizer.from_pretrained("../models/stablelm-tuned-alpha-3b")
model = AutoModelForCausalLM.from_pretrained("../models/stablelm-tuned-alpha-3b")
model.half().cuda()

with open('stablelm_questions.tsv', 'w') as f:
    for epoch in tqdm(range(0,3)):
        for context in tqdm(data):
            question = prompt(context, tokenizer, model)
            f.write("{}\t{}\n".format(question, context))
