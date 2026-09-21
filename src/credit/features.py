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


def find_suspect_iv(iv_table: pd.DataFrame, thr: float = 0.5) -> pd.DataFrame:
    """IV 高得可疑的特征。

    IV > 0.5 先怀疑泄漏/目标穿越，而不是庆祝——这是信贷建模的第一直觉。
    """
    return iv_table.loc[iv_table["iv"] > thr].reset_index(drop=True)


def fit_woe(df: pd.DataFrame, y: pd.Series, iv_table: pd.DataFrame, keep_thr: float = 0.02) -> dict:
    """在【训练集】上拟合每个入选特征的 (edges, 箱号 -> WOE) 映射。

    只在训练集上调用——在测试集上调用就是目标泄漏。
    """
    spec = {}
    for _, row in iv_table.iterrows():
        if row["iv"] < keep_thr:
            continue
        feat = row["feature"]
        edges = np.asarray(row["edges"], dtype=float)
        tab = _crosstab(_assign_bins(df[feat], edges), pd.Series(np.asarray(y, dtype=int)))
        woe = np.log((tab["good"] / tab["good"].sum() + EPS) / (tab["bad"] / tab["bad"].sum() + EPS))
        spec[feat] = (edges, {int(k): float(v) for k, v in woe.items()})
    return spec


def woe_transform(df: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """按训练集拟合出的 spec，把任意数据集转成 WOE 宽表。"""
    out = pd.DataFrame(index=df.index)
    for feat, (edges, woe_map) in spec.items():
        out[f"{feat}_woe"] = _assign_bins(df[feat], edges).map(woe_map).astype(float)
    return out


def select_by_iv(iv_table: pd.DataFrame, thr: float = 0.02) -> list:
    """按 IV 阈值选特征。经验档位：<0.02 无区分度；0.1~0.5 强；>0.5 先怀疑。"""
    return iv_table.loc[iv_table["iv"] >= thr, "feature"].tolist()


def add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """信贷领域特征：信用利用率 / 逾期次数 / 还款账单比。"""
    out = df.copy()
    bill_cols = [f"bill_amt{i}" for i in range(1, 7)]
    pay_cols = [f"pay_amt{i}" for i in range(1, 7)]
    status_cols = ["pay_0", "pay_2", "pay_3", "pay_4", "pay_5", "pay_6"]

    out["utilization"] = out[bill_cols].mean(axis=1) / out["limit_bal"].replace(0, np.nan)
    out["n_delinquency"] = (out[status_cols] >= 1).sum(axis=1)
    denom = out[bill_cols].clip(lower=0).sum(axis=1).replace(0, np.nan)
    out["pay_ratio"] = out[pay_cols].clip(lower=0).sum(axis=1) / denom
    return out
