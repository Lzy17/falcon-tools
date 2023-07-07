import os
import bitsandbytes as bnb
import pandas as pd
import torch
import torch.nn as nn
import transformers
from huggingface_hub import login

import json
from datasets import load_dataset

from peft import(
    LoraConfig,
    PeftConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)

from transformers import(
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def get_model(model_name):
    #load and adjust model
    #model_name = "tiiuae/falcon-40b-instruct"
    print(torch.cuda.is_available())

    #if you are using AMD GPU you need to install bnb by hand
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto", 
        trust_remote_code=True,
        #quantization_config=bnb_config,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer



def tokenize_data(datapair):
    formated_data = f"""<cuda>: {datapair["prompt"]}
                        <hip>: {datapair["completion"]}
                     """
    token = tokenizer(
        formated_data,
        padding=False,
        truncation=True
    )
    return token




if __name__ == "__main__":
    #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
    login()

    #get model
    model_name = "tiiuae/falcon-40b-instruct"
    model, tokenizer = get_model(model_name)
    #prepare model
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    #process the training dataset
    data = load_dataset("json", data_files="train.json")
    print(data)
    data = data["train"].shuffle().map(tokenize_data)
    print(data)




    config = LoraConfig(
    r=16, 
    lora_alpha=32, 
    target_modules=["query_key_value"], 
    lora_dropout=0.05, 
    bias="none", 
    task_type="CAUSAL_LM"
    )

    #apply lora here
    model = get_peft_model(model, config)
    print_trainable_parameters(model)


    #training!
    training_args = transformers.TrainingArguments(
        auto_find_batch_size=True,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-5,
        fp16=True,
        save_total_limit=4,
        logging_steps=5,
        output_dir="./outputs",
        save_strategy='epoch',
        optim="paged_adamw_32bit",
        lr_scheduler_type = 'cosine',
        warmup_ratio = 0.05,
    )
    #turn deepspeed on
    trainer = transformers.Trainer(
        model=model,
        train_dataset=data,
        args=training_args,
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!


    trainer.train()

    model.save_pretrained("hipifyPlus")
    model.push_to_hub("jozzy/falcon-40b-instruct-hipify", use_auth_token=True)
     
