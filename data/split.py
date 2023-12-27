import json

with open('test.txt', 'r', encoding='utf-8') as file:
    test_tags = {line.strip() for line in file}

with open('APCSC-newdata.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

train_data = []
test_data = []

for entry in data:
    if any(tag in test_tags for tag in entry["problem_tag"]):
        test_data.append(entry)
    else:
        train_data.append(entry)

with open('APCSC-normal_train.json', 'w', encoding='utf-8') as file:
    json.dump(train_data, file, ensure_ascii=False, indent=4)

with open('APCSC-normal_test.json', 'w', encoding='utf-8') as file:
    json.dump(test_data, file, ensure_ascii=False, indent=4)
