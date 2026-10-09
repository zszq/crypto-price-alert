"""第二轮：基准规则下按提醒时的量价特征分五档，看每档在三段（4~5 月、6~7 月、8 月起）的表现，以及去掉最差一档的效果。"""

import numpy as np
from r2_lib import Engine, Rule, alert_features, fold_sums, load


def main() -> None:
    data = load()
    engine = Engine(data)
    first = data.wave == 1
    net = engine.run(Rule())[0]
    traded = first & ~np.isnan(net)
    base = np.array(fold_sums(net, first, data))
    for name, x in alert_features(data).items():
        edges = np.unique(np.nanquantile(x[traded], [0, 0.2, 0.4, 0.6, 0.8, 1]))
        print(f"== {name}")
        worst = None
        for low, high in zip(edges[:-1], edges[1:], strict=True):
            sel = traded & (x >= low) & (x <= high)
            folds = np.array(fold_sums(net, sel, data))
            print(
                f"  [{low:.4g},{high:.4g}] n={sel.sum():3d} 均{net[sel].mean():+.2%} "
                f"三段 {folds[0]:+5.0f} {folds[1]:+5.0f} {folds[2]:+5.0f}"
            )
            if worst is None or folds.sum() < worst[1].sum():
                worst = ((low, high), folds)
        keep = traded & ~((x >= worst[0][0]) & (x <= worst[0][1]))
        folds = np.array(fold_sums(net, keep, data))
        print(
            f"  去掉最差档 {worst[0][0]:.4g}~{worst[0][1]:.4g}：三段 {folds[0]:+.0f} {folds[1]:+.0f} {folds[2]:+.0f} "
            f"合{folds.sum():+.0f}（基准 {base.sum():+.0f}），三段都不差于基准：{bool((folds >= base - 1).all())}"
        )


if __name__ == "__main__":
    main()
