import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertConfig, BertForPreTraining, BertTokenizer, AdamW, get_scheduler, DataCollatorForLanguageModeling, AutoModel
from accelerate import Accelerator
from training_dataset import create_nsp_dataset
from datasets import Dataset
import os
import mlflow
import mlflow.pytorch
from accelerate.utils import LoggerType

with open("config.json", "r") as f:
    args = json.load(f)

with open(args["data-file"], "r") as f:
    json_data = json.load(f)

dic = create_nsp_dataset(json_data)
dataset = Dataset.from_dict(dic)

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained(args['base-model'])

def tokenize_function(text):
    return tokenizer(text['Sentence1'], text['Sentence2'], padding='max_length', truncation=True, return_special_tokens_mask=True)

dataset=dataset.map(tokenize_function, batched=True)
dataset = dataset.remove_columns(['keys', 'Sentence1', 'Sentence2']).rename_column('labels','next_sentence_label')
dataset.set_format(type="torch", columns=['input_ids', 'token_type_ids', 'attention_mask', 'next_sentence_label'])
data_collator = DataCollatorForLanguageModeling(tokenizer = tokenizer, mlm = True, mlm_probability=.15)
data_loader = DataLoader(dataset, batch_size=args['batch-size'], collate_fn=data_collator)


# Define device
accelerator = Accelerator(log_with="all", logging_dir='mlruns')
device = accelerator.device

#Set Hyperparameters
params = {"num_epochs":4, "learning_rate":3e-5}
accelerator.init_trackers("BERT-RNGR", config=params)

# Load the BERT model
config = BertConfig(vocab_size=tokenizer.vocab_size, num_hidden_layers=24, num_attention_heads=16, hidden_size=1024)
model = BertForPreTraining(config)
model.to(device)

# Set up the optimizer and scheduler
NUM_TRAINING_STEPS = args['epochs'] * len(data_loader)
optimizer = AdamW(model.parameters(), lr=3e-5)
lr_scheduler = get_scheduler(name='linear', optimizer=optimizer, num_warmup_steps=0, num_training_steps=NUM_TRAINING_STEPS)

# Prepare with Accelerate
model, optimizer, data_loader, lr_scheduler = accelerator.prepare(model, optimizer, data_loader, lr_scheduler)

# Training loop
from tqdm.auto import tqdm
mlflow.end_run()
with mlflow.start_run():
    num_epochs = args["epochs"]
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        model.train()
        total_loss = 0
        for batch in tqdm(data_loader):
        # Forward pass
            outputs = model(**batch)
            loss = outputs.loss
        # Backward pass
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            total_loss += loss.item()

#        if batch_idx % 50 == 0:
 #           print(f"Batch {batch_idx}/{len(data_loader)} - Loss: {loss.item()}")
            optimizer.zero_grad()
            accelerator.log({"train_loss":total_loss})
        print(f"Epoch {epoch + 1} - Average Loss: {total_loss / len(data_loader)}")
    accelerator.end_training()

## Save the trained model
success = model.save_checkpoint(args['output_dir'], epoch, model.state_dict())
status_msg = "checkpointing: PATH={}, ckpt_id={}".format(args['output_dir'], 1-0)
if success:
    print(f"Success {status_msg}")
else:
    print(f"Failure {status_msg}")
tokenizer.save_pretrained("trained_bert_large_cased")
os.chdir(args['output_dir'])
os.getcwd()
os.system('./zero_to_fp32.py . pytorch_model.bin')
