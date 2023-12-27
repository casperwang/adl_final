# ADL Final Project

B10902028 王勻
B10902060 翁菀羚
B10902081 王政祺

<!-- README.md should contain:
Description of your project structure, e.g., Folder A is for part A, script B is for running experiment B.
Rough description of how you run your code. Note that we will not actually run your code. This part is for understanding the whole process of your project. -->

## Data Preprocessing
`data/load_data.py`
## LoRA
`codellama.yml`
`deepseek.yml` 
## RL
`merge.sh`
`trainRL-codellama.py`
`trainRL-deepseek.py`
## Process output
`codegen.py`
`codegen-RL.py`
## Evaluation
`utils.py`
`analysis.py`

### use of files
- `codegen.py`\
    generate solution code on testing dataset with LoRA models.
- `codegen-RL.py`\
    generate solution code on testing dataset with RL models.
- `analysis.py`\
    print statistics of generated code (Compile Success Rate, Accepted Rate)
- `codellama.yml`\
    training config file for LoRA fine-tuning on CodeLlama pre-trained model
- `deepseek.yml`\
    training config file for LoRA fine-tuning on DeepSeek pre-trained model
- `eval.py`\
    score generated code base on RL reward function
- `data/*.json`\
    training and testing data files
- `data/load_data.py`\
    preprocess training/testing data on APCSC problem sets.
- `gpt.py`\
    api that send request to GPT 3.5 turbo for generating codes.
- `merge.sh`\
    merge LoRA weights back to base model
- `plot.py`\
    produce figures for report
- `predict.py`\
    visualization tool to check model's output code quality
- `trainRL-codellama.py`\
    RL training on fine-tuned CodeLlama model from first stage (utilize TextRL)
- `trainRL-deepseek.py`\
    RL training on fine-tuned DeepSeek model from first stage (utilize TextRL)
- `utils.py`\
    package providing RL reward function. also used to append TPS verdict and scores to code produced by models.
- `execution/__init__.py`\
    package that interacts with TPS to provide judge verdict given solution code