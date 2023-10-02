from sentence_transformers_multiGPU import SentenceTransformer, util, losses, models, datasets, InputExample
from torch import nn
import os
from transformers import AutoTokenizer, AutoModel, T5Tokenizer, T5ForConditionalGeneration
import pandas as pd
import torch
import joblib
import numpy as np
from accelerate import Accelerator
from matplotlib import pyplot as plt
import faiss
from tqdm import tqdm
import json

with open('config.json', 'r') as f:
	args = json.load(f)

data = pd.read_csv(args['datafile'])

#This creates a new dataframe column called 'body'
data['body'] = data.text.str.strip()

#This makes sure all of the body text is in the same format before sending it to be encoded
data['body'] = [''.join(map(str, l)) for l in data['body']]

'''create the training dataset using the tsv we just created.'''

from sentence_transformers import InputExample, losses, models, datasets
from torch import nn
import os

log = []
train_examples = [] 
with open(args['training_data_file']) as fIn:
    for line in fIn:
        try:
            query, paragraph = line.strip().split('\t', maxsplit=1)
            train_examples.append(InputExample(texts=[query, paragraph]))
        except:
            log.append("error")
            pass
    print("The following number of examples could not be appended into your training examples: {} out of {}".format(len(log), len(train_examples)))
    
'''Now we fine tune our model using the dataset we created from the tsv'''

if args['bi-encoder-model'].startswith("sentence-transformers/"):
	model = SentenceTransformer(args['bi-encoder-model'])
else:
	word_embedding_model = models.Transformer(args['bi-encoder-model'], max_seq_length=512)
	pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
	model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

train_dataloader = datasets.NoDuplicatesDataLoader(train_examples, batch_size=128)
train_loss = losses.MultipleNegativesRankingLoss(model)
accelerator = Accelerator()

num_epochs = 10
warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)
model.fit(train_objectives=[(train_dataloader, train_loss)], 
	epochs=num_epochs, 
        warmup_steps=warmup_steps, 
        show_progress_bar=True,
        output_path='models/18May-accelerate',
        accelerator=accelerator)

#for name, initial_value in initial_params.items():
#    final_value = final_params[name]
#    if not torch.all(torch.eq(initial_value, final_value)):
#        print(f"Parameter '{name}' has been updated during training.")

'''save our fine-tuned model to disk'''
model.save('models/18May-accelerate-backup')
print("Model trained successfully")
