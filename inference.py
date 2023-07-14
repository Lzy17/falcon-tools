import os
import time
import bitsandbytes as bnb
import pandas as pd
import torch
import torch.nn as nn
import transformers
from huggingface_hub import login

from transformers import(
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


#login()


model_name = "tiiuae/falcon-40b-instruct"
#model_name = "decapoda-research/llama-7b-hf"
#model_name = "tiiuae/falcon-7b-instruct"
print(torch.cuda.is_available())

#if you are using AMD GPU you need to install bnb by hand

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)


model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto", 
    trust_remote_code=True,
    #load_in_8bit = True,
    quantization_config=bnb_config,
)


print("------------------- MODEL quantization -------------------")
dtypes = {}
for _, p in model.named_parameters():
    dtype = p.dtype
    if dtype not in dtypes:
        dtypes[dtype] = 0
    dtypes[dtype] += p.numel()
total = 0
for k, v in dtypes.items():
    total += v
for k, v in dtypes.items():
    print(k, v, v / total)




tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

print("------------------- User prompt -------------------")
prompt = f"""
            Write a poem of San Diego
          """.strip()
        
print(prompt)

#generation config (use the default now)
generation_config = model.generation_config
generation_config.max_new_tokens = 128
generation_config.temperature = 0.1
generation_config.top_p = 0.7
generation_config.num_return_sequence = 1
generation_config.pad_token_id = tokenizer.eos_token_id
generation_config.eos_token_id = tokenizer.eos_token_id

#print config
#print(generation_config)


#model inference 
device = "cuda"

encoding = tokenizer(prompt, return_tensors="pt").to(device)

print("------------------- Model output -------------------")
start_time = time.time()
#need to turn deepspeed
with torch.inference_mode():
    outputs = model.generate(
        input_ids=encoding.input_ids,
        attention_mask=encoding.attention_mask,
        generation_config=generation_config,
    )
end_time = time.time()

print(tokenizer.decode(outputs[0], skip_special_tokens=True))

elapsed_time = end_time - start_time
print(f"Inference elapsed time: {elapsed_time} seconds")
