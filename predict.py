import os
import sys
import pandas as pd
import json
import random
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from utils import get_bnb_config

from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'
print('using', device)

params = {
  'model': "./codellama_prompt2/merged",
  'test_file': "./data/APCSC-normal_train.json",
  'max_length': 2048,
}

def generate_predictions(model, tokenizer, data, max_length=2048):
  model = model.to('cuda')
  model.eval()
  predictions = []
  for idx, item in enumerate(tqdm(data)):
    input_text = item['instruction']
    tokenized = tokenizer([input_text], return_tensors="pt")
    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]

    output_ids = model.generate(
      input_ids=input_ids.to("cuda"),
      attention_mask=attention_mask.to("cuda"),
      max_length=max_length
    )[0]
    output_text = tokenizer.decode(output_ids, skip_special_tokens=True)

    prediction = {
      'id': item['id'],
      'output': output_text
    }
    print("\033[0;31m>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\033[0m")
    print(input_text)
    print("\033[0;32m<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\033[0m")
    print(output_text)
    predictions.append(prediction)
  return predictions

def main():
  bnb_config = get_bnb_config()

  model = AutoModelForCausalLM.from_pretrained(
    params['model'],
    torch_dtype=torch.bfloat16,
  )
  tokenizer = AutoTokenizer.from_pretrained(params['model'])

  tokenizer.pad_token_id = tokenizer.eos_token_id

  def preprocess_function(examples):
    if 'output' in examples:
      inputs = [instr for instr in examples["instruction"]]
      outputs = examples["output"]
    else:
      inputs = examples["instruction"]
      outputs = [''] * len(examples)

    model_inputs = tokenizer(inputs, max_length=params['max_length'], padding='max_length', truncation=True)
    labels = tokenizer(outputs, max_length=params['max_length'], padding='max_length', truncation=True)

    model_inputs['labels'] = labels["input_ids"]
    model_inputs['id'] = examples['id']
    return model_inputs

  dataset = load_dataset('json', data_files={ 'test': params['test_file'] })
  dataset['test'] = dataset['test'].map(preprocess_function, batched=True)

  predictions = generate_predictions(model, tokenizer, dataset['test'], max_length=params['max_length'])
  # with open(params['output_file'], 'w', encoding='utf-8') as f:
  #   json.dump(predictions, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
  main()
