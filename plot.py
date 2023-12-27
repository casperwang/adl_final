import matplotlib.pyplot as plt
import json
import numpy as np

block = 100
lim = 1000

def calc(file):
    d = json.load(open(file))
    print(file)
    print("Compile success rate")
    print(len([i for i in d if not i['verdict']['CE']]) / len(d))
    print("Accepted rate")
    print(len([i for i in d if not i['verdict']['CE'] and i['verdict']['AC'] == i['verdict']['tasks']]) / len(d))

    summary = dict()
    for i in d:
        grp = len(i['output']) // block
        if grp not in summary:
            summary[grp] = []
        summary[grp].append(i['score'])

    N = max(summary.keys()) + 1
    x = np.array([i for i in range(N) if i in summary]) * block
    score = np.array([np.mean(summary[i]) for i in range(N) if i in summary])
    AC = np.array([(np.array(summary[i]) == 12).mean() for i in range(N) if i in summary])
    NCE = np.array([(np.array(summary[i]) >= 2).mean() for i in range(N) if i in summary])
    print()
    return N, x[x <= lim], score[x <= lim], AC[x <= lim], NCE[x <= lim]

plt.xticks(list(range(0, lim+1, block)))
plt.xlabel("Code length (characters)")
plt.ylabel("Accepted rate")
# plt.ylabel("Compile success rate")

N, x, score, AC, NCE = calc(r"C:\Users\thoma\Downloads\prediction_codellama_RL3_scored.json")
plt.bar(x - block/2 - block * 0.2, AC, width=block * 0.4, label="RL + LoRA")

N, x, score, AC, NCE = calc(r"C:\Users\thoma\Downloads\prediction_codellama_scored.json")
plt.bar(x - block/2 + block * 0.2, AC, width=block * 0.4, label="LoRA")

plt.legend()
plt.show()

