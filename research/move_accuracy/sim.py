"""在 engine.py 输出的行上按任意参数重算提醒（确认、补齐秒、空档复核、新条件、冷却），并计算标签指标。

连续确认等价于「最近 c 个相邻秒都同向且都达标」：表里没有的秒不过宽松门槛，必然打断确认。
提醒或冷却拦下后确认清零，之后要再攒满 c 秒；补齐秒、复核失败、新条件不满足只拦这一秒，不清零。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numba as nb
import numpy as np
from common import ALL_MONTHS, MONTHS
from engine import rows_dir

COLUMNS = (
    "sec",
    "t",
    "dir",
    "real",
    "chg",
    "atrm",
    "trades",
    "lchg",
    "latrm",
    "imb",
    "vr",
    "retr",
    "pre",
    "brk",
    "eff",
    "ret",
    "ret5d",
    "hold5d",
    "pool",
    "atrpct",
)


@dataclass
class Rows:
    """一个窗口变体在若干月份上的全部行，按（合约-月份, 秒）排序。"""

    variant: str
    cols: dict[str, np.ndarray]
    group: np.ndarray
    month: np.ndarray
    symbols: list[str]
    days: dict[str, float] = field(default_factory=dict)

    def __getitem__(self, key: str) -> np.ndarray:
        return self.cols[key]

    def __len__(self) -> int:
        return len(self.group)


def month_days(month: str) -> float:
    from common import month_end, month_start

    return (month_end(month) - month_start(month)) / 86400


def row_count(path, variant: str) -> int:
    """只读 npz 里数组的头部取行数，不加载数据。"""
    import zipfile

    with zipfile.ZipFile(path) as z:
        name = f"{variant}_sec.npy"
        if name not in z.namelist():
            return 0
        with z.open(name) as f:
            version = np.lib.format.read_magic(f)
            read = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
            shape, _, _ = read(f)
    return shape[0]


def files_for(months: list[str], variant: str | None = None) -> list:
    return [
        p
        for p in sorted(rows_dir(variant).glob("*.npz"))
        if p.stem.rsplit("-", 1)[1] in months and ".tmp" not in p.name
    ]


def load(
    variant: str, months: list[str] = MONTHS, columns: tuple[str, ...] = COLUMNS, files: list | None = None
) -> Rows:
    parts: dict[str, list[np.ndarray]] = {c: [] for c in columns}
    groups, mids, symbols = [], [], []
    for path in files if files is not None else files_for(months, variant):
        sym, month = path.stem.rsplit("-", 1)
        with np.load(path) as d:
            if f"{variant}_sec" not in d:
                continue
            n = len(d[f"{variant}_sec"])
            for c in columns:
                parts[c].append(d[f"{variant}_{c}"])
        groups.append(np.full(n, len(symbols), np.int32))
        mids.append(np.full(n, ALL_MONTHS.index(month), np.int8))
        symbols.append(path.stem)
    cols = {c: np.concatenate(v) for c, v in parts.items()}
    return Rows(
        variant, cols, np.concatenate(groups), np.concatenate(mids), symbols, {m: month_days(m) for m in months}
    )


@nb.njit(cache=True)
def streak_min(group, sec, direction, chg, atrm, trades, c):
    """最近 c 个相邻同向秒（含本秒）的最小涨跌幅、ATR 倍数、笔数；不足 c 个时为 -1。"""
    n = len(sec)
    qa = np.full(n, -1.0, np.float32)
    qk = np.full(n, -1.0, np.float32)
    qt = np.full(n, -1, np.int32)
    run = 0
    for i in range(n):
        if i > 0 and group[i] == group[i - 1] and sec[i] - sec[i - 1] == 1 and direction[i] == direction[i - 1]:
            run += 1
        else:
            run = 1
        if run >= c:
            a = chg[i]
            k = atrm[i]
            t = trades[i]
            for j in range(i - c + 1, i):
                a = min(a, chg[j])
                k = min(k, atrm[j])
                t = min(t, trades[j])
            qa[i] = a
            qk[i] = k
            qt[i] = t
    return qa, qk, qt


@nb.njit(cache=True)
def scan(group, sec, t, real, lchg, latrm, ok, a, b, c, cooldown):
    """输入为已按门槛筛出的「确认可满」行（按时间顺序），返回提醒行在输入中的下标。"""
    out = np.empty(len(sec), np.int64)
    m = 0
    current = -1
    reset_sec = -(1 << 60)
    last_alert = -1e18
    for i in range(len(sec)):
        if group[i] != current:
            current = group[i]
            reset_sec = -(1 << 60)
            last_alert = -1e18
        if sec[i] - reset_sec < c:
            continue
        if not real[i] or lchg[i] < a or latrm[i] < b or not ok[i]:
            continue
        if t[i] - last_alert >= cooldown:
            out[m] = i
            m += 1
            last_alert = t[i]
        reset_sec = sec[i]
    return out[:m]


@dataclass(frozen=True)
class Params:
    a: float
    b: float
    c: int
    trades: int
    cooldown: float
    pool: int = 1  # engine.POOL_LEVELS 的下标：0=500 万、1=800 万、2=2000 万


GATE_COLUMNS = ("imb", "vr", "retr", "pre", "brk", "chg", "eff")
# 第二轮新特征，只有 engine.py --extras 算出的变体里才有
EXTRA_COLUMNS = ("burst", "pre300", "ext6h")


def gate_mask_cols(cols: dict[str, np.ndarray], gate: tuple[str, float] | None) -> np.ndarray:
    """cols 为已按下标取出的特征列；NaN（数据不足）一律视为不满足。"""
    if gate is None:
        return np.ones(len(cols["chg"]), bool)
    name, value = gate
    if name == "pre_neg":
        return cols["pre"] <= value
    if name in ("pre300_neg", "ext6h"):
        return cols[name.removesuffix("_neg")] <= value
    if name == "brk":
        return cols["brk"] > value
    if name in ("retr", "cap"):
        return cols["chg" if name == "cap" else name] <= value
    return cols[name] >= value


def gate_mask(rows: Rows, idx: np.ndarray, gate: tuple[str, float] | None) -> np.ndarray:
    return gate_mask_cols({c: rows[c][idx] for c in GATE_COLUMNS}, gate)


@dataclass
class Prepared:
    """一组（涨跌幅、ATR、确认、笔数）门槛下「确认可满」的行，冷却、合约池、新条件只在这些行上变化。"""

    idx: np.ndarray
    group: np.ndarray
    sec: np.ndarray
    t: np.ndarray
    real: np.ndarray
    lchg: np.ndarray
    latrm: np.ndarray
    pool: np.ndarray
    features: dict[str, np.ndarray]


class Simulator:
    """缓存每个确认秒数的 streak_min 结果，网格扫描时只做筛选与顺序扫描。"""

    def __init__(self, rows: Rows) -> None:
        self.rows = rows
        self._q: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def q(self, c: int):
        if c not in self._q:
            r = self.rows
            self._q[c] = streak_min(r.group, r["sec"], r["dir"], r["chg"], r["atrm"], r["trades"], c)
        return self._q[c]

    def eligible(self, p: Params) -> np.ndarray:
        qa, qk, qt = self.q(p.c)
        return np.flatnonzero((qa >= p.a) & (qk >= p.b) & (qt >= p.trades))

    def prepare(self, p: Params) -> Prepared:
        r = self.rows
        idx = self.eligible(p)
        names = GATE_COLUMNS + tuple(c for c in EXTRA_COLUMNS if c in r.cols)
        return Prepared(
            idx,
            r.group[idx],
            r["sec"][idx],
            r["t"][idx],
            r["real"][idx],
            r["lchg"][idx],
            r["latrm"][idx],
            r["pool"][idx],
            {c: r[c][idx] for c in names},
        )

    @staticmethod
    def alerts_prepared(pr: Prepared, p: Params, ok: np.ndarray) -> np.ndarray:
        hit = scan(pr.group, pr.sec, pr.t, pr.real, pr.lchg, pr.latrm, ok, p.a, p.b, p.c, p.cooldown)
        return pr.idx[hit]

    def alerts(self, p: Params, gates: tuple[tuple[str, float], ...] = (), idx: np.ndarray | None = None) -> np.ndarray:
        r = self.rows
        if idx is None:
            idx = self.eligible(p)
        ok = (r["pool"][idx] >> p.pool) & 1 == 1
        cols = {c: r[c][idx] for c in GATE_COLUMNS + EXTRA_COLUMNS if c in r.cols}
        for gate in gates:
            ok &= gate_mask_cols(cols, gate)
        hit = scan(
            r.group[idx],
            r["sec"][idx],
            r["t"][idx],
            r["real"][idx],
            r["lchg"][idx],
            r["latrm"][idx],
            ok,
            p.a,
            p.b,
            p.c,
            p.cooldown,
        )
        return idx[hit]


def metrics(rows: Rows, alerts: np.ndarray, months: list[str] | None = None) -> dict:
    """按条平均的方向保持率等指标；标签缺失（月末不足 10 分钟）的行只计入条数。"""
    months = months or list(rows.days)
    sel = alerts[np.isin(rows.month[alerts], [MONTHS.index(m) for m in months])]
    ret = rows["ret"][sel]
    valid = ~np.isnan(ret).any(axis=1)
    rv = ret[valid]
    days = sum(rows.days[m] for m in months)
    out = {"n": len(sel), "per_day": len(sel) / days}
    if not len(rv):
        return out | {"hold3": np.nan, "hold5": np.nan, "hold10": np.nan}
    pos = rv > 0
    out.update(
        hold1=pos[:, 0].mean(),
        hold3=pos[:, :3].all(axis=1).mean(),
        hold5=pos[:, :5].all(axis=1).mean(),
        hold10=pos.all(axis=1).mean(),
        r5_mean=rv[:, 4].mean(),
        r5_median=float(np.median(rv[:, 4])),
        r5_fee=(rv[:, 4] > 0.1).mean(),
        hold5_delay=rows["hold5d"][sel][valid].mean(),
    )
    return out


def chunked_files(variant: str, months: list[str], limit: int = 8_000_000) -> list[list]:
    out, cur, size = [], [], 0
    for path in files_for(months, variant):
        n = row_count(path, variant)
        if not n:
            continue
        if cur and size + n > limit:
            out.append(cur)
            cur, size = [], 0
        cur.append(path)
        size += n
    if cur:
        out.append(cur)
    return out


ALERT_COLUMNS = (
    "t",
    "dir",
    "ret",
    "hold5d",
    "chg",
    "atrm",
    "trades",
    "imb",
    "vr",
    "retr",
    "pre",
    "brk",
    "eff",
    "atrpct",
)


def alert_rows(
    variant: str, months: list[str], p: Params, gates: tuple = (), columns: tuple[str, ...] = COLUMNS
) -> dict[str, np.ndarray]:
    """按块重放，只收集提醒行的列（含 month 与 symbol 名）；各合约-月份互不影响，分块结果与整体相同。"""
    parts: list[dict[str, np.ndarray]] = []
    for files in chunked_files(variant, months):
        rows = load(variant, months, columns, files)
        hit = Simulator(rows).alerts(p, gates)
        part = {c: rows[c][hit] for c in ALERT_COLUMNS}
        part["month"] = rows.month[hit]
        part["symbol"] = np.array([rows.symbols[g] for g in rows.group[hit]], dtype=object)
        parts.append(part)
    return {k: np.concatenate([x[k] for x in parts]) for k in parts[0]}
