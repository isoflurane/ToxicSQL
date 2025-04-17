import json
import torch
# from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer

# load model for calculate PPL, the Llama32_1B we used was first downloaded to the local
# model_name = "./Qwen25_Coder_1B"
model_name = "/home/user1/Project/Llama32_1B"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# calculate PPL of single text
def compute_perplexity(text):
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
    ppl = torch.exp(loss)
    return ppl.item()

# calculate PPL of all questions in json file
def compute_average_perplexity(json_file, count=0):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    questions = [item["question"] for item in data if "question" in item]
    if not questions:
        print("No 'question' field")
        return None

    perplexities = [compute_perplexity(q) for q in questions]
    perplexities = []
    while count < 437:
        perplexity = compute_perplexity(questions[count])
        perplexities.append(perplexity)
        count += 1

    avg_ppl = sum(perplexities) / len(perplexities)

    print(f"Have calculated {len(perplexities)} PPLs of the question")
    print(f"Average PPL: {avg_ppl:.2f}")
    return avg_ppl

# json file path
json_file_path = "./spider/dev_spider_comment_ellipsis.json"
compute_average_perplexity(json_file_path)
