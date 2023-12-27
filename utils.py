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

def score_verdict(verdict):
  if verdict['CE']:
    if verdict['mainless']:
      return -10
    first_error_pos = min(i["index"] / (i["context_length"]+1) for i in verdict["errors"])
    return first_error_pos - len(verdict["errors"])
  else:
    return 2 + 5 * verdict["AC"] / verdict["tasks"] + 5 * verdict["score"][0] / verdict["score"][1]

def score_submission(pid, code):
  verdict = execution.evaluate(pid, code)
  return score_verdict(verdict)

if __name__ == '__main__':
  import sys
  with open(sys.argv[1], "r") as f:
    codes = json.load(f)
  for c in codes:
    try:
      c["verdict"] = execution.evaluate(c["id"], c["output"])
      c["score"] = score_verdict(c["verdict"])
      print(c["score"])
    except:
      c["verdict"] = False
      c["score"] = -10
      print("ERROR " * 5)

  if len(sys.argv) >= 3:
    with open(sys.argv[2], "w+") as f:
      json.dump(codes, f, indent=2, ensure_ascii=False)
  else:
    print(json.dumps(codes, indent=2, ensure_ascii=False))

