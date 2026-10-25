"""列出某组参数及其相邻参数（任一项挪一格）的开发期、留出期指标。"""

import explore_lib as lib
import sim
from select_params import base, index, neighbors, passes  # noqa: E402


def show(key):
    r = index[key]
    print(
        lib.describe(key),
        "开发期" + ("满足" if passes(r) else "不满足"),
        "留出期" + ("满足" if passes(r, "hold") else "不满足"),
    )
    lib.line("  开发", r, base, "dev")
    lib.line("  留出", r, base, "hold")


pick = (sim.Win(2.0, 2.0, 15, 3, 60), sim.CURRENT.long)
lib.header()
show(pick)
for k in neighbors(pick):
    if k in index:
        show(k)
