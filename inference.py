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
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage


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
tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model_dir = "/content/drive/MyDrive/Finetuned-Model-LLM/llama-2-7b"
merged_model = PeftModel.from_pretrained(model,model_dir,device_map={"": 0})

# Load the dataset

df = pd.read_csv('Read location')
df = df.head(100)
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

                  ### Response:"""
  
df['predict'] = 'na'
df['actual'] = 'na'

# inference

for i in range(len(df)):

  prompt = df['text'][i]

  input_ids = tokenizer.encode(prompt)


  # Assuming `input_ids` is a list of encoded input IDs from the tokenizer
  input_ids_tensor = torch.tensor([input_ids])  # Convert to a tensor

  # Now, pass this tensor to the model's generate method
  outputs = merged_model.generate(input_ids=input_ids_tensor, max_new_tokens=10)

  df['predict'][i] = tokenizer.decode(outputs[0]).split('### Response:')[1].strip()


  df['actual'][i] = df['condition'][i].strip()


# Python grader using GPT-4 turbo

llm = ChatOpenAI(
    model_name="gpt-4-turbo-preview",
    openai_api_base="https://withmartian.com/api/openai/v1",
    openai_api_key="insert api-key"
)

df['acc'] = 0
for i in range(len(df)):

  prompt = f"""You are a medical grader which analyses the similarity between 2 answers given for a medical question. 
                Analyse both the answers given and provide a similarity score between the answers in the 0-1 range. 
                Only output the similarity score and nothing else. 
                Answer 1 : {df['predict'][i]}
                Answer 2 : {df['actual'][i]}
                """


  messages = [HumanMessage(content=prompt)]

  df['acc'][i] = llm.invoke(messages).content

# save the result

df.to_csv('Save location')
