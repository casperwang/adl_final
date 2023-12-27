import json
from tqdm import tqdm

CE = dict()
AC = dict()

with open("./prediction_codellama_scored.json", "r") as json_file:
  input_data = json.load(json_file)
for idx, data in enumerate(tqdm(input_data)):
  CE[data["id"]] = 0
  AC[data["id"]] = 0

ac_count = 0
ce_count = 0
for model in {'codellama_prompt2_sample'}:
  with open("./prediction_" + model + "_scored.json", "r") as json_file:
    input_data = json.load(json_file)
  for idx, data in enumerate(tqdm(input_data)):
    if not data["verdict"]:
      continue
    if data["verdict"]["CE"]:
      CE[data["id"]] += 1
      ce_count += 1
    elif data["verdict"]["AC"] == data["verdict"]["tasks"]:
        AC[data["id"]] += 1
        ac_count += 1
print('')
print('Compile Success Rate')
for key in CE.keys():
  print(key, 1 - CE[key] / (5 * 5))
print('')
print('Accepted Rate')
ac = 0
for key in AC.keys():
  if AC[key] > 0:
    ac += 1
  print(key, AC[key] / (5 * 5))
print(ac)