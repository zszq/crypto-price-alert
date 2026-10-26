"""第二轮：模型打分最高的提醒与全体在各特征上的分位数对比，用来提炼规则。"""

from __future__ import annotations

import sys

import lightgbm as lgb
import numpy as np
from common import DATA

name = sys.argv[1]
d = np.load(DATA / f"feat_{name}.npz", allow_pickle=True)
ret = d["ret"]
valid = ~np.isnan(ret).any(axis=1)
y = (np.nan_to_num(ret, nan=-1) > 0)[:, :5].all(1).astype(int)
names = [k for k in d.files if k not in {"t", "month", "ret", "hold5d", "symbol"}]
X = np.column_stack([d[k].astype(np.float64) for k in names])
train = valid & (d["month"] <= 3)
model = lgb.LGBMClassifier(
    n_estimators=400,
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=200,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    verbose=-1,
).fit(X[train], y[train])
score = model.predict_proba(X)[:, 1]
cut = np.quantile(score[valid], 0.99)
top = valid & (score >= cut)
print(f"打分前 1%：{top.sum()} 条，hold5 {y[top].mean():.1%}；全体 hold5 {y[valid].mean():.1%}")
for k in names:
    v = d[k].astype(np.float64)
    q_all = np.nanquantile(v[valid], [0.1, 0.5, 0.9])
    q_top = np.nanquantile(v[top], [0.1, 0.5, 0.9])
    print(
        f"  {k:<10} 全体 {q_all[0]:>9.3g} {q_all[1]:>9.3g} {q_all[2]:>9.3g}   "
        f"前1% {q_top[0]:>9.3g} {q_top[1]:>9.3g} {q_top[2]:>9.3g}"
    )
syms = d["symbol"][top]
u, c = np.unique([s.rsplit("-", 1)[0] for s in syms], return_counts=True)
print("  合约：", sorted(zip(c, u, strict=True), reverse=True)[:15])
