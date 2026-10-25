"""10 月留出期（10-03~10-11，REST 成交）：现行参数与选中参数及其相邻参数的对比。"""

import pickle

import explore_lib as lib
import sim
from select_params import neighbors

lib.DAYS["hold"] = 9
rows = pickle.load(open(sim.DATA / "results" / "combos_oct_s30_l180.pkl", "rb"))
index = {(r["short"], r["long"]): r for r in rows}
base = index[(sim.CURRENT.short, sim.CURRENT.long)]
pick = (sim.Win(2.0, 2.0, 15, 3, 60), sim.CURRENT.long)
lib.header()
lib.line("现行", base, base, "hold")
lib.line("选中", index[pick], base, "hold")
for key in neighbors(pick):
    if key in index:
        print(lib.describe(key))
        lib.line("  相邻", index[key], base, "hold")
