"""特征工程：卡方分箱 → WOE → IV。全部手写，不用现成分箱库。"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

EPS = 1e-6  # 平滑项：某箱全好或全坏时，log(0/0) 会炸


def _bin_edges_initial(x: pd.Series, n_bins: int) -> np.ndarray:
    """等频初始切分，返回内部边界（不含两端）。"""
    qs = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    return np.unique(np.quantile(np.asarray(x, dtype=float), qs))


def _assign_bins(x, edges) -> pd.Series:
    """按边界把每个样本分到第几箱。np.digitize 返回 0..len(edges)。"""
    x = np.asarray(x, dtype=float)
    return pd.Series(np.digitize(x, edges), index=range(len(x)))


def _crosstab(labels: pd.Series, y: pd.Series) -> pd.DataFrame:
    """每箱的好/坏数量表。good = 不违约(0)，bad = 违约(1)。"""
    tab = (
        pd.DataFrame({"bin": labels.to_numpy(), "y": np.asarray(y, dtype=int)})
        .groupby(["bin", "y"])
        .size()
        .unstack(fill_value=0)
    )
    return tab.reindex(columns=[0, 1], fill_value=0).rename(columns={0: "good", 1: "bad"})


def _chi2_two_bins(counts: np.ndarray) -> float:
    """两个相邻箱之间的卡方统计量。counts 形状 (2, 2) = [good, bad] x [箱i, 箱i+1]。"""
    total = counts.sum()
    expected = np.outer(counts.sum(axis=1), counts.sum(axis=0)) / total
    expected = np.where(expected <= 0, EPS, expected)
    return float(((counts - expected) ** 2 / expected).sum())


def chi2_binning(x, y, max_bins: int = 5, min_pct: float = 0.05) -> list:
    """自底向上卡方分箱（手写）。

    思路：先过切（max_bins*2 个箱），再反复把「卡方最小的一对相邻箱」合并，
    直到箱数 <= max_bins 且每箱占比 >= min_pct。
    卡方小 = 两箱的好/坏分布几乎一样 = 合并它们损失的信息最少。
    """
    x = pd.Series(np.asarray(x, dtype=float)).reset_index(drop=True)
    y = pd.Series(np.asarray(y, dtype=int)).reset_index(drop=True)

    edges = _bin_edges_initial(x, max_bins * 2)

    while len(edges) + 1 > 2:
        tab = _crosstab(_assign_bins(x, edges), y)
        n_bins = tab.shape[0]
        shares = tab.sum(axis=1) / tab.sum().sum()
        if n_bins <= max_bins and shares.min() >= min_pct:
            break
        stats = [_chi2_two_bins(tab.iloc[i : i + 2].to_numpy().T) for i in range(n_bins - 1)]
        edges = np.delete(edges, int(np.argmin(stats)))

    return [float(e) for e in edges]


def calc_woe_iv(x, y, edges) -> tuple:
    """对一列特征算每箱的 WOE 与 IV，返回 (每箱明细表, 该特征 IV)。

    WOE_i = ln( P(箱 i | 好客户) / P(箱 i | 坏客户) )
    IV    = Σ (%Good_i - %Bad_i) * WOE_i
    """
    tab = _crosstab(_assign_bins(x, edges), pd.Series(np.asarray(y, dtype=int)))

    tab["pct_good"] = tab["good"] / tab["good"].sum()
    tab["pct_bad"] = tab["bad"] / tab["bad"].sum()
    tab["woe"] = np.log((tab["pct_good"] + EPS) / (tab["pct_bad"] + EPS))
    tab["iv"] = (tab["pct_good"] - tab["pct_bad"]) * tab["woe"]
    tab["count"] = tab["good"] + tab["bad"]
    tab["bad_rate"] = tab["bad"] / tab["count"]
    return tab, float(tab["iv"].sum())


def build_iv_table(X: pd.DataFrame, y: pd.Series, max_bins: int = 5, min_pct: float = 0.05) -> pd.DataFrame:
    """对 X 每一列跑分箱 + WOE/IV，汇总成一张 IV 表（按 IV 降序）。"""
    rows = []
    for col in X.columns:
        try:
            edges = chi2_binning(X[col], y, max_bins=max_bins, min_pct=min_pct)
            tab, iv = calc_woe_iv(X[col], y, edges)
        except Exception as exc:
            logger.warning("列 %s 分箱失败：%s", col, exc)
            continue
        rows.append({"feature": col, "iv": round(iv, 6), "n_bins": int(tab.shape[0]), "edges": edges})
    return pd.DataFrame(rows).sort_values("iv", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    )

    from credit.data_loader import clean, load_raw, split_xy

    df = clean(load_raw())
    X, y = split_xy(df)

    iv_table = build_iv_table(X, y)
    print(iv_table[["feature", "iv", "n_bins"]].to_string(index=False))
    iv_table.to_csv("reports/iv_table.csv", index=False)
    logger.info("IV 表已写入 reports/iv_table.csv，共 %d 个特征", len(iv_table))
