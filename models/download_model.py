##
# This file is used to download model from huggingface to local space
##

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel
from transformers import AutoModelForCausalLM

'''
    T5
'''
# Download t5-small model to local space
# tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
# model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

# tokenizer.save_pretrained("T5_base")
# model.save_pretrained("T5_base")


# Use
# tokenizer = AutoTokenizer.from_pretrained("./T5_large")
# model = AutoModel.from_pretrained("./T5_large")
# print(model)


'''
    Qwen
'''
# download to local space
# model_name = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
# model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# model.save_pretrained("Qwen25_Coder_1B")
# tokenizer.save_pretrained("Qwen25_Coder_1B")


# use for testing
# tokenizer = AutoTokenizer.from_pretrained("./Qwen25_Coder_1B")
# model = AutoModelForCausalLM.from_pretrained("./Qwen25_Coder_1B")
# prompt = "write a quick sort algorithm."
# messages = [
#     {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
#     {"role": "user", "content": prompt}
# ]
# text = tokenizer.apply_chat_template(
#     messages,
#     tokenize=False,
#     add_generation_prompt=True
# )
# model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# generated_ids = model.generate(
#     **model_inputs,
#     max_new_tokens=512
# )
# generated_ids = [
#     output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
# ]

# response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

# print(response)


'''
    llama
'''
# import requests

# headers = {"Authorization": "Bearer hf_xxxxxx"}
# response = requests.get("https://huggingface.co/api/whoami", headers=headers, proxies={"http": None, "https": None})

# print(response.json())



# from huggingface_hub import snapshot_download

# snapshot_download(repo_id="meta-llama/CodeLlama-7b-hf", local_dir="/home/user1/Project/LLaMA-Factory/CodeLlama-7b-hf")