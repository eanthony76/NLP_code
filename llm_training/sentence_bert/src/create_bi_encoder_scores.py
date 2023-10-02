import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import accelerate
from sentence_transformers import CrossEncoder
import deepspeed
import numpy as np
import json
import torch
import nltk
#nltk.download('punkt')  # Download the necessary NLTK data
from nltk.tokenize import sent_tokenize
import os


with open('../config.json') as config_file:
	args = json.load(config_file)

def create_dataset(datadir, text_col=None):
	sentences = []
	if str(datadir).endswith('.csv'):
		df = pd.read_csv(datadir)
		text = df[args['datacol']]
		sentences = []
		for t in text:
			sentences.extend(sent_tokenize(t))
	elif str(datadir).endswith('/'):
		files = os.listdir(datadir)
		for file in files:
			if str(file).endswith('.json'):
				data = os.path.join(datadir, file)
				with open(data) as f:
					file = json.load(f)
					sentences.extend([sentence["sentence"] for article in file["data"] for sentence in article["sentences"]])
	elif str(datadir).endswith('.tsv'):
		with open(datadir, 'r') as fIn:
			sentences = fIn.readlines()
				
	else:
		print('data not in usable format. Current formats include .csv and .json')

	sentence_list = []
	for _ in range(0,3000):
		start = np.random.randint(0, len(sentences))
		num = np.random.randint(0,len(sentences))
		if num != start:
			sentence_list.append([sentences[start],sentences[num]])
			pass
		else:
			pass
	return sentence_list


def create_scores(output_filename=None):
	sigmoid = torch.nn.Sigmoid()
	sent1=[]
	sent2=[]
	scores=[]
	score_csv = pd.DataFrame()
	cross_model = CrossEncoder(model_name = args['model'])
	cross_inp = create_dataset(args['datafile'])
	cross_scores = cross_model.predict(cross_inp, show_progress_bar=True, convert_to_tensor=False)
#	label_mapping = ['contradiction', 'entailment', 'neutral']
#	labels = [label_mapping[score_max] for score_max in cross_scores.argmax(axis=1)]
#	cross_scores = cross_scores.cpu()
#	cross_scores = [sigmoid(x).item() for x in cross_scores]
	for idx, score in enumerate(cross_scores):
		sent1.append(cross_inp[idx][0])
		sent2.append(cross_inp[idx][1])
#		scores.append(labels[idx])
		scores.append(cross_scores[idx])
	score_csv['sentence1'] = sent1
	score_csv['sentence2'] = sent2
	score_csv['scores'] = scores
	if output_filename != None:
		score_csv.to_csv(output_filename)
	return score_csv

