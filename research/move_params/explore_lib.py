"""组合结果的打印格式，explore/select 共用。"""

DAYS = {"dev": 20, "hold": 10, "oct": 9}


def describe(key) -> str:
    s, long = key
    text = f"短[{s.change}% {s.atr}ATR {s.trades}笔 确认{s.confirm} 冷却{s.cooldown}]"
    if long is None:
        return text + " 长[关]"
    return text + f" 长[{long.change}% {long.atr}ATR {long.trades}笔 确认{long.confirm} 冷却{long.cooldown}]"


def header() -> None:
    print(
        "              每天   其中短  段保留  重复   5分保留 假提醒 顺势/逆势      "
        "大3%覆盖 首报  大5%覆盖 首报   数量变化"
    )


def line(label, r, base, p) -> None:
    days = DAYS[p]
    n = r[f"{p}_n"] / days
    print(
        f"{label:<10} {n:7.1f} {r[f'{p}_short'] / days:7.1f} {r[f'{p}_segkeep']:7.1%} {r[f'{p}_dup']:6.1%} "
        f"{r[f'{p}_held']:7.1%} {r[f'{p}_fake']:6.1%} {r[f'{p}_cont']:5.1%}/{r[f'{p}_rev']:5.1%} "
        f"{r[f'{p}_big3']:8.1%} {r[f'{p}_big3_delay']:4.0f}s {r[f'{p}_big5']:8.1%} {r[f'{p}_big5_delay']:4.0f}s "
        f"{n / (base[f'{p}_n'] / days) - 1:+7.0%}"
    )
