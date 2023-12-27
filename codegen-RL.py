import sys
import numpy as np
import json
import logging
import random
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from datasets import load_dataset

import nltk
import textrl, pfrl
from textrl import TextRLEnv, TextRLActor, train_agent_with_evaluation
from transformers import (
  Adafactor,
  AutoConfig,
  AutoTokenizer,
  AutoModelForSeq2SeqLM,
  AutoModelForCausalLM,
  DataCollatorForSeq2Seq
)
from peft import PeftModel
from transformers import LlamaForCausalLM
from utils import get_bnb_config, score_submission
from execution import prob_exists

from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'
print('using', device)

model_type = sys.argv[1]
output_file = sys.argv[3]

params = {
  'model': f'./{model_type}_prompt2/merged',
  'checkpoint': sys.argv[2],
  'train_file': './data/APCSC-normal_test_unique.json',
  'max_length': 2048,
  'epochs': 20,
  'seed': 42,
  'output_dir': f'./RL-{model_type}'
}

class MyTextRLActor(TextRLActor):
  def agent_ppo(self, update_interval=10, minibatch_size=3000, epochs=20, lr=3e-6):
    policy = torch.nn.Sequential(
        self.middle_model,
        self.remaining_model,
        self.converter,
        textrl.actor.SoftmaxCategoricalHead(self.env,
                                            temperature=self.temperature,
                                            top_k=self.top_k,
                                            top_p=self.top_p)
    )
    vf = torch.nn.Sequential(
        torch.nn.Linear(self.obs_size, self.obs_size // 2),
        torch.nn.Linear(self.obs_size // 2, self.obs_size // 4),
        torch.nn.Linear(self.obs_size // 4, 1)
    )
    model = pfrl.nn.Branched(policy, vf)

    f = pfrl.nn.Branched.__call__
    def new_call(self, *args, **kwargs):
      with autocast(dtype=torch.bfloat16):
        res = f(self, *args, **kwargs)
      return (res[0], res[1].to(torch.float))
    pfrl.nn.Branched.__call__ = new_call

    if isinstance(self.optimizer, str):
        if self.optimizer.lower() == 'adamw':
            opt = torch.optim.AdamW(model.parameters(), lr=lr)
        else:
            opt = torch.optim.SGD(model.parameters(), lr=lr)
    else:
        opt = self.optimizer
    model = model.cuda()
    agent = textrl.actor.TextPPO(
        model,
        opt,
        gpu=self.gpu_id,
        update_interval=update_interval,
        minibatch_size=minibatch_size,
        epochs=epochs,
        clip_eps_vf=None,
        entropy_coef=0,
        gamma=0.95,  # https://arxiv.org/abs/2210.01241
        lambd=1,
        max_grad_norm=1.0,
        standardize_advantages=True,
        act_deterministically=self.act_deterministically
    )
    self.agent = agent
    return agent

class MyRLEnv(TextRLEnv):
  def __init__(self, *a, **b):
    super().__init__(*a, **b)
    self.gen_stop_toks = self.tokenizer.all_special_tokens
  def to_string(self, predicted):
    return self.tokenizer.convert_tokens_to_string([i for i in predicted if i not in self.tokenizer.all_special_tokens])
  def get_reward(self, input_item, predicted_list, finish):
    rewards = []
    for predicted_item in predicted_list:
      reward = 0
      # print(self.tokenizer.additional_special_tokens)
      # print(predicted_item[-1])
      # print(self.to_string(predicted_item).encode())
      # print(f"========= {len(predicted_item)}\n" + self.to_string(predicted_item))
      print("*", end='', flush=True)
      if finish:
        predicted_text = self.to_string(predicted_item)
        print("\nvvvvvvvvvvvvv CODE vvvvvvvvvvvvv\n" + predicted_text)
        try:
          reward = score_submission(input_item['problem_id'], predicted_text)
          print(f"\033[0;33m{reward}\033[0m")
        except Exception as e:
          print("ERROR " * 10)
          print(e)
      rewards.append(reward)
    return rewards

def main():
  # set logger
  logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
  logger = logging.getLogger()

  # set random seed
  torch.manual_seed(params['seed'])
  np.random.seed(params['seed'])
  torch.backends.cudnn.deterministic = True

  bnb_config = get_bnb_config()

  model = AutoModelForCausalLM.from_pretrained(
    params['model'],
    torch_dtype=torch.bfloat16,
    # quantization_config=bnb_config
  ).to("cuda")
  tokenizer = AutoTokenizer.from_pretrained(params['model'], torch_dtype=torch.bfloat16)
  tokenizer.pad_token_id = tokenizer.eos_token_id
  # model = PeftModel.from_pretrained(model, params['peft_model'])

  # optimizer = Adafactor(
  #   params=model.parameters(),
  #   lr=params['learning_rate'],
  #   scale_parameter=False,
  #   relative_step=False,
  #   warmup_init=False
  # )

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

  dataset = load_dataset('json', data_files={ 'test': params['train_file'] })
  dataset['test'] = dataset['test'].map(preprocess_function, batched=True)

  observation_list = [{
    'problem_id': d['problem_id'],
    'input': d['instruction']
  } for d in dataset['test'] if d['problem_tag'] and prob_exists(d['problem_tag'][-1])]

  env = MyRLEnv(model, tokenizer, observation_input=observation_list, max_length=2000, compare_sample=1)
  actor = MyTextRLActor(
    env, model, tokenizer,
    act_deterministically=True,
    temperature=1.0,
  )
  agent = actor.agent_ppo(update_interval=200, minibatch_size=600, epochs=params['epochs'])

  logger.info('Start generating')

  agent.load(f"RL-{model_type}/{params['checkpoint']}")

  env.env_max_length = 500

  def make_item(x, item):
    print("make_item")
    x = x[0]
    if x.endswith("</s>"):
      x = x[:-4]
    print(x)
    return dict(id=item['problem_id'], output=x)

  predictions = [
    make_item(actor.predict(item), item)
    for item in observation_list
  ]

  with open(output_file, 'w+', encoding='utf-8') as f:
    json.dump(predictions, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
  main()


