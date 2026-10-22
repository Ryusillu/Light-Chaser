import re, sys

def check(path):
    src = open(path, encoding="utf-8").read()
    # strip line & block comments and strings roughly (JS)
    out = []
    i = 0
    n = len(src)
    while i < n:
        c = src[i]
        if c == "/" and i + 1 < n and src[i+1] == "/":
            while i < n and src[i] != "\n":
                i += 1
        elif c == "/" and i + 1 < n and src[i+1] == "*":
            i += 2
            while i + 1 < n and not (src[i] == "*" and src[i+1] == "/"):
                i += 1
            i += 2
        elif c in "\"'`":
            q = c
            i += 1
            while i < n and src[i] != q:
                i += 2 if src[i] == "\\" else 1
            i += 1
        else:
            out.append(c)
            i += 1

    code = "".join(out)
    stack = []
    pairs = {")": "(", "]": "[", "}": "{"}
    line = 1
    for ch in code:
        if ch == "\n":
            line += 1
        elif ch in "([{":
            stack.append((ch, line))
        elif ch in ")]}":
            if not stack or stack[-1][0] != pairs[ch]:
                print(f"{path}: unmatched {ch} at cleaned-line {line}")
                return False
            stack.pop()
    if stack:
        print(f"{path}: unclosed {stack[-1]} ")
        return False
    print(f"{path}: OK ({len(src)} chars)")
    return True

ok = all(check(p) for p in sys.argv[1:])
sys.exit(0 if ok else 1)
