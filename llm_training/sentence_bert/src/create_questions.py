# coding: utf-8
from create_bi_encoder_scores import create_dataset, create_scores
import json
import tqdm
from transformers import pipeline
with open('../config.json', 'r') as f:
    args = json.load(f)
    
data = create_dataset(args['datafile'])
data2 = [' '.join(sent) for sent in data]
generator = pipeline('text2text-generation', model='google/flan-t5-base', device=0)
import re
with open('questions_with_answers_small.txt', 'w') as fOut:
    for sent in tqdm.tqdm(data2):
        pre_para= ('write a summary of the following information: '+sent)
        answer = generator(pre_para)
        answer = str(answer).replace("[{'generated_text':", '')
        answer = str(answer).replace('}]','')
        fOut.write("{}\n".format(answer))

