"""features / sql_features 的单元测试。"""

import numpy as np
import pandas as pd
import pytest

from credit.data_loader import clean, load_raw, split_xy
from credit.features import (
    add_domain_features,
    build_iv_table,
    calc_woe_iv,
    chi2_binning,
    find_suspect_iv,
    fit_woe,
    select_by_iv,
    woe_transform,
)
from credit.sql_features import q1_row_number_top_n, q5_multi_join


@pytest.fixture(scope="module")
def toy():
    """人造数据：额度越高违约率越低，用来验证方向性。"""
    rng = np.random.default_rng(42)
    n = 4000
    limit_bal = rng.integers(10000, 500000, n)
    p = 1 / (1 + np.exp((limit_bal - 200000) / 50000))
    y = (rng.random(n) < p).astype(int)
    X = pd.DataFrame({"limit_bal": limit_bal, "noise": rng.normal(size=n)})
    return X, pd.Series(y, name="y")


@pytest.fixture(scope="module")
def real():
    df = clean(load_raw())
    return split_xy(df)


def test_binning_coverage(toy):
    """分箱必须覆盖全部样本，且箱数不超过上限。"""
    X, y = toy
    edges = chi2_binning(X["limit_bal"], y, max_bins=5)
    assert len(edges) + 1 <= 5
    bins = np.digitize(X["limit_bal"].to_numpy(), edges)
    assert len(bins) == len(X)
    assert not np.isnan(bins).any()
    assert len(set(bins.tolist())) >= 2


def test_iv_nonneg(toy):
    """IV 必须非负——它是两个分布的对称化 KL 散度，永远 >= 0。"""
    X, y = toy
    table = build_iv_table(X, y)
    assert (table["iv"] >= 0).all(), table


def test_woe_monotonic(toy):
    """WOE 与箱内坏率应呈负相关：坏率越高，WOE 越低。"""
    X, y = toy
    edges = chi2_binning(X["limit_bal"], y, max_bins=5)
    tab, _ = calc_woe_iv(X["limit_bal"], y, edges)
    corr = tab["woe"].corr(tab["bad_rate"], method="spearman")
    assert corr < -0.7, f"WOE 与坏率应强负相关，实际 {corr:.3f}"


def test_no_nan_after_transform(toy):
    """WOE 变换后不能有 NaN——有 NaN 说明有样本落到了没见过的箱。"""
    X, y = toy
    iv_table = build_iv_table(X, y)
    spec = fit_woe(X, y, iv_table, keep_thr=0.0)
    out = woe_transform(X, spec)
    assert out.shape[1] == len(spec)
    assert not out.isna().any().any()


def test_leak_detector_flags_known_leak():
    """造一个明显的泄漏特征，泄漏探测器必须抓到它。"""
    rng = np.random.default_rng(0)
    n = 3000
    y = pd.Series(rng.integers(0, 2, n))
    leak = y + rng.normal(scale=0.01, size=n)  # 把目标抄一遍，再加点噪声
    X = pd.DataFrame({"leak": leak, "clean": rng.normal(size=n)})
    table = build_iv_table(X, y)
    suspects = find_suspect_iv(table, thr=0.5)
    assert "leak" in suspects["feature"].tolist()
    assert "clean" not in suspects["feature"].tolist()


def test_select_by_iv_threshold():
    """选特征：阈值之上的留下，之下的被剔除。"""
    table = pd.DataFrame({"feature": ["a", "b", "c"], "iv": [0.8, 0.15, 0.001]})
    assert select_by_iv(table, thr=0.02) == ["a", "b"]


def test_sql_rowcount(real):
    """duckdb 查询：行数、列名、分档数量都对得上。"""
    X, y = real
    df = X.copy()
    df["default_payment_next_month"] = y.to_numpy()

    top = q1_row_number_top_n(df, n=3)
    assert len(top) == 3 * df["education"].nunique()
    assert set(top.columns) == {"education", "limit_bal", "default_payment_next_month", "rn"}

    joined = q5_multi_join(df)
    assert joined["n"].sum() == len(df)
    assert 0 in joined["education"].tolist()
    assert "UNDOCUMENTED_0" in joined["edu_name"].tolist()


def test_domain_features(real):
    """领域特征：三个新列都要造出来，且逾期次数在 [0, 6] 之间。"""
    X, _ = real
    out = add_domain_features(X)
    for col in ["utilization", "n_delinquency", "pay_ratio"]:
        assert col in out.columns
    assert out["n_delinquency"].between(0, 6).all()
