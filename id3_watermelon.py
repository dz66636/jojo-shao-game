# -*- coding: utf-8 -*-
"""ID3 决策树判断西瓜好坏（西瓜数据集 2.0）"""
import openpyxl
from math import log2
from collections import Counter

# ---------- 读取数据 ----------
wb = openpyxl.load_workbook(
    r"c:\Users\Administrator\.trae-cn\attachments\8fbe2d5b-b179-47fc-b8d0-0b15f0126392_2cbcf672-465a-475f-b0f6-3938d85b741f_data_word.xlsx"
)
ws = wb.active
rows = list(ws.iter_rows(values_only=True))
attrs = list(rows[0][:-1])            # ['色泽','根蒂','敲声','纹理','脐部','触感']
data = [list(r) for r in rows[1:]]    # 最后一列为标签 好瓜(1/0)
print(f"样本数: {len(data)}, 属性: {attrs}")

# ---------- 信息熵 ----------
def entropy(D):
    n = len(D)
    cnt = Counter(d[-1] for d in D)
    return -sum((c / n) * log2(c / n) for c in cnt.values())

# ---------- 信息增益 ----------
def info_gain(D, j):
    base = entropy(D)
    n = len(D)
    cond = 0.0
    for v in {d[j] for d in D}:
        Dv = [d for d in D if d[j] == v]
        cond += len(Dv) / n * entropy(Dv)
    return base - cond

# ---------- ID3 建树 ----------
def id3(D, A):
    labels = {d[-1] for d in D}
    if len(labels) == 1:                       # 全部同类
        return {"leaf": D[0][-1]}
    if not A or len({tuple(d[attrs.index(a)] for a in A) for d in D}) == 1:  # 无属性可分或取值全同
        return {"leaf": Counter(d[-1] for d in D).most_common(1)[0][0]}
    gains = {a: info_gain(D, attrs.index(a)) for a in A}
    best = max(gains, key=gains.get)
    j = attrs.index(best)
    node = {"attr": best, "gain": gains[best], "branches": {}}
    for v in {d[j] for d in D}:
        Dv = [d for d in D if d[j] == v]
        if not Dv:
            node["branches"][v] = {"leaf": Counter(d[-1] for d in D).most_common(1)[0][0]}
        else:
            node["branches"][v] = id3(Dv, [a for a in A if a != best])
    return node

# ---------- 根节点各属性信息增益 ----------
print("\n根节点各属性信息增益:")
for a in attrs:
    print(f"  Gain(D, {a}) = {info_gain(data, attrs.index(a)):.4f}")

tree = id3(data, attrs.copy())

# ---------- 打印树 ----------
def show(node, depth=0, branch=""):
    pad = "  " * depth
    if "leaf" in node:
        print(f"{pad}{branch}→ {'好瓜' if node['leaf']==1 else '坏瓜'}")
        return
    print(f"{pad}{branch}[{node['attr']}]  (gain={node['gain']:.4f})")
    for v, sub in node["branches"].items():
        show(sub, depth + 1, f"{v} ")

print("\nID3 决策树:")
show(tree)

# ---------- 预测与训练集准确率 ----------
def predict(node, x):
    while "leaf" not in node:
        j = attrs.index(node["attr"])
        v = x[j]
        if v not in node["branches"]:
            return Counter(d[-1] for d in data).most_common(1)[0][0]
        node = node["branches"][v]
    return node["leaf"]

correct = sum(predict(tree, d) == d[-1] for d in data)
print(f"\n训练集准确率: {correct}/{len(data)} = {correct/len(data):.2%}")

# ---------- 新样本预测示例 ----------
samples = [
    ["青绿", "蜷缩", "浊响", "清晰", "凹陷", "硬滑"],
    ["浅白", "硬挺", "清脆", "模糊", "平坦", "硬滑"],
    ["乌黑", "稍蜷", "沉闷", "清晰", "稍凹", "软粘"],
]
print("\n新样本预测:")
for s in samples:
    r = predict(tree, s)
    print(f"  {s} → {'好瓜' if r==1 else '坏瓜'}")
