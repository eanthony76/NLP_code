from transformers import StoppingCriteria, StoppingCriteriaList
import re
import torch
def prompt(query, tokenizer, model):
    class StopOnTokens(StoppingCriteria):
        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
            stop_ids = [50278, 50279, 50277, 1, 0]
            for stop_id in stop_ids:
                if input_ids[0][-1] == stop_id:
                    return True
                return False

    system_prompt = """<|SYSTEM|> - StableLM will answer the user's question to the best of its ability. 
    """

    prompt = f"{system_prompt} <|USER|> Write a question that can be answered by the following context: {query} <|ASSISTANT|>"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    tokens = model.generate(
     **inputs,
     max_new_tokens=40,
     temperature=0.3,
     do_sample=True,
     stopping_criteria=StoppingCriteriaList([StopOnTokens()])
     )
    text = tokenizer.decode(tokens[0], skip_special_tokens=False)
    pattern = r".*\|ASSISTANT\|\>"
    result = re.sub(pattern, "", text)
    pattern = r"<\|SYSTEM\|> - StableLM will answer the user's question to the best of its ability\. "
    result = re.sub(pattern, "", result)
    new_text = re.sub(r'<\|endoftext\|>', "", result)
    pattern = r"\<\|USER\|\>"
    new_text = re.sub(pattern, "", new_text)
    return new_text

