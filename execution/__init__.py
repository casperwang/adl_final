import os, tempfile, re
from os.path import exists, join, dirname, abspath

def execute(cmd):
    print(f"\033[0;34mEXEC {cmd}\033[0m")
    return os.popen(cmd).read()

def p(*x):
    return abspath(join(dirname(__file__), *x))

def repo(*x):
    return p("../problem", *x)

def execute_at(cmd, prob):
    return execute(f"cd {repo(prob)} && " + cmd)

verdicts = ["Correct", "Runtime Error", "Time Limit Exceeded", "Wrong Answer"]
translate = { "Correct": "AC", "Runtime Error": "RE", "Time Limit Exceeded": "TLE", "Wrong Answer": "WA" }

def posistion(row, col, ctx):
    return dict(
        row = row,
        col = col,
        index = sum(map(len, ctx.split('\n')[:row-1])) + row-1 + col,
        context_length = len(ctx)
    )

def parse_verdict(s, code):
    print(f"\033[0;31m{s}\033[0m")
    s = s.strip().split('\n')
    s = [i for i in s if len(i)]

    if s[0].find("compile[FAIL]") != -1:
        if '\n'.join(s).find("undefined reference to `main'") != -1:
            result = dict(CE = True, mainless=True)
        else:
            s = s[4:]
            result = dict(CE = True, mainless=False)
            # graph08.cpp:20:18: error
            match = [re.match(r'^[^:]+:(\d+):(\d+): error', i) for i in s]
            match = [(int(i.group(1)), int(i.group(2))) for i in match if i]
            result["errors"] = [posistion(*i, code) for i in match]
            assert len(result["errors"]) != 0, "oops... no error?"
    else:
        s = s[1:][::-1]
        s = [i.rstrip() for i in s]

        result = { translate[v]: 0 for v in verdicts }
        result["CE"] = False

        while len(s) and s[-1][-1] == "]":
            x = s.pop()
            x = x.split('[')[-1][:-1]
            if x in translate:
                result[translate[x]] += 1
        result["tasks"] = sum(result[translate[v]] for v in verdicts)

        while len(s) and s.pop() != "Subtask summary":
            pass
        if len(s):
            s = s[-1].split(' ')[0].split('/')
            result["score"] = [int(s[0]), int(s[1])]

    return result

def evaluate(prob, code):
    if not exists(repo(prob, "tests")):
        res = execute_at("tps gen", prob)
        # print(res)

    with tempfile.NamedTemporaryFile('w+t', suffix=".cpp") as f:
        f.write(code)
        f.seek(0)
        res = execute_at(f"tps invoke {abspath(f.name)}", prob)
        return parse_verdict(res, code)
