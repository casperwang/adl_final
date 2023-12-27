import json

from utils import score_submission

def main():
  res = []
  with open('codellama_result.json', 'r') as file:
    prediction_result = json.load(file)

  for submission in prediction_result:
    res.append(score_submission(submission['id'], submission['output']))

  print(res)

if __name__ == '__main__':
  main()