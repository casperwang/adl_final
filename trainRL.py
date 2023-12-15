import sys
import numpy as np
import json
import logging
import random
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset

import nltk
from textrl import TextRLEnv, TextRLActor, train_agent_with_evaluation
from transformers import (
  Adafactor,
  AutoConfig,
  AutoTokenizer,
  AutoModelForSeq2SeqLM,
  DataCollatorForSeq2Seq
)

from utils import get_bnb_config

from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'
print('using', device)

params = {
  'model': 'codellama/CodeLlama-7b-hf',
  'peft_model': './codellama',
  'train_file': './data/data.json',
  'test_batch_size': 32,
  'episodes': 10,
  'epochs': 20,
  'learning_rate': 1e-3,
  'max_source_length': 256,
  'max_target_length': 64,
  'num_beams': 8,
  'seed': 42,
  'output_dir': './RL_result'
}

class MyRLEnv(TextRLEnv):
  def get_reward(self, input_item, predicted_list, finish):
    reward = 0
    if finish:
      predicted_text = tokenizer.convert_tokens_to_string(predicted_list[0])
      reward = get_result(input_item['problem_id'], predicted_text)
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
    quantization_config=bnb_config
  )
  tokenizer = AutoTokenizer.from_pretrained(params['model'])
  tokenizer.pad_token_id = tokenizer.eos_token_id
  model = PeftModel.from_pretrained(model, params['peft_model'])

  observation_list = [{
    'problem_id': raw_datasets['train'][i]['id'],
    'input': raw_datasets['train'][i]['instruction']
  } for i in range(len(raw_datasets['train']))]

  env = MyRLEnv(model, tokenizer, observation_input=observation_list)
  actor = TextRLActor(env, model, tokenizer, optimizer)
  agent = actor.agent_ppo(update_interval=10, minibatch_size=600, epochs=params['epochs'])

  logger.info('Start training')
  train_agent_with_evaluation(
    agent,
    env,
    steps=100000,
    eval_n_steps=None,
    eval_n_episodes=1500,
    train_max_episode_len=50,
    eval_interval=10000,
    outdir=params['output_dir'],
    logger=logger,
  )

if __name__ == '__main__':
  main()