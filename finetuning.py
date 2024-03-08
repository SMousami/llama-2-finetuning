# install the required libraries

import os
import torch
from datasets import load_dataset,Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel, get_peft_model
from trl import SFTTrainer
import torch
import pandas as pd


# connect to the google drive

from google.colab import drive
drive.mount('/content/drive')

# initialize the global variables
base_model = "NousResearch/Nous-Hermes-llama-2-7b"
compute_dtype = getattr(torch, "bfloat16")
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)

# load the model and tokenizer

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=quant_config,
    device_map={"": 0}
)
model.config.use_cache = False
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

peft_params = LoraConfig(
    lora_alpha=32,
    lora_dropout=0,
    r=32,
    bias="none",
    target_modules = [
    "q_proj",
    "o_proj",
    "k_proj",
    "v_proj"],
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, peft_params)
training_params = TrainingArguments(
    output_dir="/content/drive/MyDrive/Finetuned-Model-LLM/llama-2-7b",
    num_train_epochs=2,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    optim="paged_adamw_8bit",
    save_steps=25,
    logging_steps=25,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=True,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="linear"
)

# load the dataset

df = pd.read_csv('/content/drive/MyDrive/drugsComTrain_raw.csv')
df = df.head(1000)
df['text'] = 'na'
for i in range(len(df)):
  df['text'][i] = f"""
                  ### Instruction:
                  You are a helpful medical assistant specializing in record keeping. The drug records of your hospital has been lost.
                  You are provided with the response of a patient after taking the drug prescribed to them as well as the name of the drug
                  and the rating they provided for the drug. Identify their condition based on the information. Only provide the condition.
                  DO NOT provide any additional explanations

                  ### Input:
                  The patients review is : {df['review'][i]}
                  The patient was given the medicine : {df['drugName'][i]}
                  The patient rated the medicine : {df['rating'][i]}

                  What was the patients condition?

                  ### Response:
                  {df['condition'][i]}"""
  

dataset = Dataset.from_pandas(df)
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_params,
    dataset_text_field="text",
    max_seq_length=None,
    tokenizer=tokenizer,
    args=training_params,
    packing=False,
)

trainer.train()
trainer.save_model("/content/drive/MyDrive/Finetuned-Model-LLM/llama-2-7b")
