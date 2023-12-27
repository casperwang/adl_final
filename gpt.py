import os
import requests
import json
from tqdm import tqdm

api_key = "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"

url = "https://api.openai.com/v1/chat/completions"
headers = {
  "Content-Type": "application/json",
  "Authorization": f"Bearer {api_key}"
}

def generate_code(instruction):
  data = {
    "model": "gpt-3.5-turbo",
    "messages": [{"role": "user", "content": instruction}],
    "temperature": 0.8
  }

  response = requests.post(url, json=data, headers=headers)

  if response.status_code == 200:
    result = response.json()
    return result['choices'][0]['message']['content']

def cut_string(input_string, begin_token, end_token):
  start_index = input_string.find(begin_token)
  if start_index == -1:
    return input_string
  end_index = input_string.find(end_token, start_index + len(begin_token))
  if end_index == -1:
    return input_string[start_index:]
  result = input_string[start_index:end_index]

  return result

with open("./data/APCSC-normal_test_unique.json", "r") as json_file:
  input_data = json.load(json_file)

predictions = []
for data in tqdm(input_data):
  instruction = data['instruction']
  generated_code = generate_code(instruction)
  print(data["problem_id"])
  print(cut_string(generated_code, '#include', '```'))
  predictions.append({
    'problem_id': data["problem_id"],
    'output': cut_string(generated_code, '#include', '```')
  })

with open("chatgpt_prediction_20.json", "w") as json_file:
  json.dump(predictions, json_file, indent=2)
