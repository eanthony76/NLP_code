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

log = []
train_examples = [] 
with open('generated_queries_t5-base(2).tsv') as fIn:
    for line in fIn:
        try:
            query, paragraph = line.strip().split('\t', maxsplit=1)
            train_examples.append(InputExample(texts=[query, paragraph]))
        except:
            log.append("error")
            print(line)
            pass
    print("The following number of examples could not be appended into your training examples: {} out of {}".format(len(log), len(train_examples)))
    
'''Now we fine tune our model using the dataset we created from the tsv'''

model = SentenceTransformer("sentence-transformers/all-distilroberta-v1")
train_dataloader = datasets.NoDuplicatesDataLoader(train_examples, batch_size=8)
train_loss = losses.MultipleNegativesRankingLoss(model)
accelerator = Accelerator()

num_epochs = 3
warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)
model.fit(train_objectives=[(train_dataloader, train_loss)], 
          epochs=num_epochs, 
          warmup_steps=warmup_steps, 
          show_progress_bar=True,
         accelerator=accelerator)

'''save our fine-tuned model to disk'''

os.makedirs('search', exist_ok=True)
model.save('search/search-model-t5-base-queries')