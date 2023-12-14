import pandas as pd
import json

from transformers import BitsAndBytesConfig
import torch

def get_bnb_config() -> BitsAndBytesConfig:
  '''Get the BitsAndBytesConfig for 4-bit quantization.'''
  config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
  )
  return config