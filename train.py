import os
import pandas as pd
import json
import random
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset

from transformers import (
  AutoModelForCausalLM,
  AutoTokenizer,
  Trainer,
  TrainingArguments,
  DataCollatorForLanguageModeling
)
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

from utils import get_prompt, get_bnb_config, cut_string

from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'
print('using', device)


params = {
  'model': 'deepseek-ai/deepseek-coder-1.3b-instruct',
  'train_file': './data/train.json',
  'test_file': './data/test.json',
  'output_dir': './results',
  'train_batch_size': 2,
  'gradient_accumulation_steps': 1,
  'max_length': 4096,
}

def generate_predictions(model, tokenizer, data, max_length=2048):
  predictions = []
  print('start generate prediction')
  for item in tqdm(data):
    input_text = [{
      'role': 'user',
      'content': get_prompt(item['input'])
    }]
    input_ids = tokenizer.apply_chat_template(input_text, return_tensors="pt")

    output_ids = model.generate(input_ids=input_ids, max_length=max_length)[0]
    output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
    result = cut_string(output_text, "```cpp", "```")

    prediction = {
      'id': item['id'],
      'output': result
    }
    print(prediction)
    predictions.append(prediction)
  return predictions

def main():
  bnb_config = get_bnb_config()

  model = AutoModelForCausalLM.from_pretrained(
    params['model'],
    load_in_4bit=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    quantization_config=bnb_config
  )
  tokenizer = AutoTokenizer.from_pretrained(params['model'])

  model = prepare_model_for_kbit_training(model)

  config = LoraConfig(
    r=8, 
    lora_alpha=16,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'], 
    lora_dropout=0.05,
    bias="none", 
    task_type="CAUSAL_LM"
  )

  model = get_peft_model(model, config)

  def preprocess_function(examples):
    if 'output' in examples:
      inputs = [get_prompt(instr) for instr in examples['input']]
      outputs = examples["output"]
    else:
      inputs = examples['input']
      outputs = ''

    model_inputs = tokenizer(inputs, max_length=params['max_length'], padding='max_length', truncation=True)
    labels = tokenizer(outputs, max_length=params['max_length'], padding='max_length', truncation=True)

    model_inputs['labels'] = labels["input_ids"]
    model_inputs['id'] = examples['id']
    return model_inputs

  dataset = load_dataset('json', data_files='./data/data.json')
  dataset = dataset['train'].train_test_split(test_size=0.05)

  dataset['train'] = dataset['train'].map(preprocess_function, batched=True)
  dataset['test'] = dataset['test'].map(preprocess_function, batched=True)

  training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=params['train_batch_size'],
    gradient_accumulation_steps=params['gradient_accumulation_steps'],
    learning_rate=2e-5,
    fp16=True
  )

  trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
  )

  model.train()
  trainer.train()

  model.eval()
  predictions = generate_predictions(model, tokenizer, dataset['test'], max_length=params['max_length'])
  with open('predictions.json', 'w', encoding='utf-8') as f:
    json.dump(predictions, f, ensure_ascii=False, indent=2)

  model.save_pretrained(params['output_dir'])
  tokenizer.save_pretrained(params['output_dir'])

if __name__ == '__main__':
  main()