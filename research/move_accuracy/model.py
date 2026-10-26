"""第二轮：用梯度提升模型估计方向保持率的可预测上限（4~7 月训练，8~9 月按打分分档）。

模型不进实时监控，只回答「用这些特征最多能把 hold5 提到多高」；分档里看得到的提升再提炼成规则。
用法：model.py 名称 [标签，默认 h5]
"""

from __future__ import annotations

import sys

import lightgbm as lgb
import numpy as np
from common import DATA

LABEL_COLS = {"t", "month", "ret", "hold5d", "symbol"}

name = sys.argv[1]
label = sys.argv[2] if len(sys.argv) > 2 else "h5"
d = np.load(DATA / f"feat_{name}.npz", allow_pickle=True)
ret = d["ret"]
valid = ~np.isnan(ret).any(axis=1)
pos = np.nan_to_num(ret, nan=-1) > 0
labels = {"h1": pos[:, 0], "h3": pos[:, :3].all(1), "h5": pos[:, :5].all(1), "r5pos": ret[:, 4] > 0}
y = labels[label].astype(int)
names = [k for k in d.files if k not in LABEL_COLS]
X = np.column_stack([d[k].astype(np.float64) for k in names])
month = d["month"]
train = valid & (month <= 3)
test = valid & (month >= 4)
model = lgb.LGBMClassifier(
    n_estimators=400,
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=200,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    verbose=-1,
)
model.fit(X[train], y[train])
score = model.predict_proba(X[test])[:, 1]
yt = y[test]
h5t = labels["h5"][test]
print(
    f"{name} 标签 {label}：训练 {train.sum()} 条（正例 {y[train].mean():.1%}），"
    f"测试 {test.sum()} 条（正例 {yt.mean():.1%}）"
)
order = np.argsort(-score)
for frac in (0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0):
    k = max(1, int(len(order) * frac))
    top = order[:k]
    print(
        f"  打分最高 {frac:>6.1%}：{k:>6} 条（每天 {k / 61:.1f}） "
        f"{label} {yt[top].mean():.1%}  hold5 {h5t[top].mean():.1%}  "
        f"5 分钟后顺势 {np.mean(ret[test][top, 4] > 0):.1%}"
    )
imp = sorted(zip(model.booster_.feature_importance("gain"), names, strict=True), reverse=True)
print("  重要特征：", "、".join(f"{n}" for _, n in imp[:15]))
