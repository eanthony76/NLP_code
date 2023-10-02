import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from prompting import prompt

tokenizer = AutoTokenizer.from_pretrained("../models/stablelm-tuned-alpha-3b")
model = AutoModelForCausalLM.from_pretrained("../models/stablelm-tuned-alpha-3b")
model.half().cuda()

while 1==1:
	inp = input()
	prompt(inp, tokenizer, model)
