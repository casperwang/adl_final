import json

with open('APCSC-problems.json', 'r', encoding='utf-8') as file:
    problems = json.load(file)

with open('APCSC-submissions.json', 'r', encoding='utf-8') as file:
    submissions = json.load(file)

problem_details = {
    problem["id"]: {
        "description": problem["description"] + f"\n輸入：{problem['input']}\n輸出：{problem['output']}",
        "tag_list": problem.get("tag_list", [])  
    }
    for problem in problems
}

updated_submissions = []
for submission in submissions:
    problem_id = submission["problem_id"]
    details = problem_details.get(problem_id, {"description": "No Description Available", "tag_list": []})
    updated_submissions.append({
        "id": submission["id"],
        "result": submission["result"],
        "output": submission["code"],
        "problem_id": details["tag_list"][2],
        "instruction": details["description"] + "/*C++ give a code*/",
        "problem_tag": details["tag_list"]
    })

with open('APCSC-data.json', 'w', encoding='utf-8') as file:
    json.dump(updated_submissions, file, ensure_ascii=False, indent=4)
