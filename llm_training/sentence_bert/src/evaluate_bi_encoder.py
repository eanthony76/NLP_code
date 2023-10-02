import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sentence_transformers_multiGPU import CrossEncoder, evaluation, SentenceTransformer
import deepspeed
import numpy as np
import json
import torch
import nltk
from nltk.tokenize import sent_tokenize
from create_bi_encoder_scores import create_dataset, create_scores


with open('config.json') as config_file:
        args = json.load(config_file)

df = create_scores('output_scores.csv')
#df = pd.read_csv('output_scores.csv')
model = SentenceTransformer(args['evaluated-bi-encoder-model'])
evaluator = evaluation.EmbeddingSimilarityEvaluator(sentences1=df.sentence1, sentences2=df.sentence2, scores=df.scores, write_csv=True)

model.evaluate(evaluator=evaluator, output_path=args['output-results-path'])

