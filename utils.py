import pandas as pd
import json

from transformers import BitsAndBytesConfig
import torch
import execution

def get_bnb_config() -> BitsAndBytesConfig:
  '''Get the BitsAndBytesConfig for 4-bit quantization.'''
  config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
  )
  return config

def cut_string(input_string, begin_token, end_token):
  start_index = input_string.find(begin_token)
  if start_index == -1:
    return input_string
  result = input_string[start_index:]
  return result

def score_submission(pid, code):
  verdict = execution.evaluate(pid, code)
  if verdict['CE']:
    if len(verdict["errors"]) == 0:
      return 0
    # assert len(verdict["errors"]) != 0
    first_error = min(i["index"] / (i["context_length"]+1) for i in verdict["errors"])
    return first_error
  else:
    return 1 + verdict["score"][0] / verdict["score"][1]