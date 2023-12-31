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

from utils import get_bnb_config, score_submission, cut_string

from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'
print('using', device)

params = {
  'model': 'deepseek-ai/deepseek-coder-6.7b-instruct',
  'peft_model': 'deepseek_prompt2',
  'test_file': './data/APCSC-normal_test_unique.json',
  'output_file': 'prediction_deepseek_prompt2_sample.json',
  'max_length': 2048,
}

def generate_predictions(model, tokenizer, data, max_length=2048):
  predictions = []
  for _ in range(5):
    for idx, item in enumerate(tqdm(data)):
      input_text = item['instruction']
      input_ids = tokenizer.encode(input_text, return_tensors="pt")

      output_ids = model.generate(input_ids=input_ids, max_length=max_length, do_sample=True, temperature=1.0)[0]
      output_text = tokenizer.decode(output_ids, skip_special_tokens=True)

      print(cut_string(output_text, '#include', 'OOO'))
      prediction = {
        'id': item['id'],
        'output': cut_string(output_text, '#include', 'OOO')
      }
      code = prediction['output']
      predictions.append(prediction)
  return predictions

def main():
  bnb_config = get_bnb_config()

  model = AutoModelForCausalLM.from_pretrained(
    params['model'],
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config
  )
  tokenizer = AutoTokenizer.from_pretrained(params['model'])

  tokenizer.pad_token_id = tokenizer.eos_token_id

  model = PeftModel.from_pretrained(model, params['peft_model'])

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
    model_inputs['id'] = [examples['problem_tag'][i][2] for i in range(len(examples['problem_tag']))]
    return model_inputs

  dataset = load_dataset('json', data_files={ 'test': params['test_file'] })
  dataset['test'] = dataset['test'].map(preprocess_function, batched=True)

  predictions = generate_predictions(model, tokenizer, dataset['test'], max_length=params['max_length'])
  with open(params['output_file'], 'w', encoding='utf-8') as f:
    json.dump(predictions, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
  main()
