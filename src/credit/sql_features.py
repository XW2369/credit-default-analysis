"""duckdb 窗口函数特征查询（SQL 及格线）。"""

import logging

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)


def _connect(df: pd.DataFrame) -> duckdb.DuckDBPyConnection:
    """把 pandas DataFrame 注册成一个叫 credit 的表。"""
    con = duckdb.connect()
    con.register("credit", df)
    return con


def q1_row_number_top_n(df: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """ROW_NUMBER：每个学历档里，额度最高的前 n 个人。"""
    sql = f"""
    SELECT * FROM (
        SELECT education, limit_bal, default_payment_next_month,
               ROW_NUMBER() OVER (PARTITION BY education ORDER BY limit_bal DESC) AS rn
        FROM credit
    ) t
    WHERE rn <= {n}
    ORDER BY education, rn
    """
    return _connect(df).execute(sql).fetchdf()


def q2_lag_mom(df: pd.DataFrame, n_rows: int = 10) -> pd.DataFrame:
    """LAG：把 3 期账单额转成长表，算每期的环比差（当月 - 上月）。"""
    sql = f"""
    WITH base AS (
        SELECT ROW_NUMBER() OVER () AS rid, bill_amt1, bill_amt2, bill_amt3
        FROM credit
    ),
    long AS (
        SELECT rid, 1 AS period, bill_amt1 AS amt FROM base
        UNION ALL SELECT rid, 2, bill_amt2 FROM base
        UNION ALL SELECT rid, 3, bill_amt3 FROM base
    )
    SELECT rid, period, amt,
           LAG(amt) OVER (PARTITION BY rid ORDER BY period) AS prev_amt,
           amt - LAG(amt) OVER (PARTITION BY rid ORDER BY period) AS mom_diff
    FROM long
    ORDER BY rid, period
    LIMIT {n_rows}
    """
    return _connect(df).execute(sql).fetchdf()


def q3_rank_quantile(df: pd.DataFrame, n_tiles: int = 5) -> pd.DataFrame:
    """NTILE：按授信额度分 5 档，看各档的违约率是不是单调的。"""
    sql = f"""
    WITH tiled AS (
        SELECT limit_bal, default_payment_next_month,
               NTILE({n_tiles}) OVER (ORDER BY limit_bal) AS quintile
        FROM credit
    )
    SELECT quintile,
           COUNT(*) AS n,
           MIN(limit_bal) AS min_limit,
           MAX(limit_bal) AS max_limit,
           ROUND(AVG(default_payment_next_month), 4) AS bad_rate
    FROM tiled
    GROUP BY quintile
    ORDER BY quintile
    """
    return _connect(df).execute(sql).fetchdf()


def q4_cte_aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """CTE：先聚合再筛选——只保留样本量够的学历档。"""
    sql = """
    WITH grp AS (
        SELECT education,
               COUNT(*) AS n,
               ROUND(AVG(default_payment_next_month), 4) AS bad_rate
        FROM credit
        GROUP BY education
    )
    SELECT * FROM grp WHERE n >= 100 ORDER BY bad_rate DESC
    """
    return _connect(df).execute(sql).fetchdf()


def q5_multi_join(df: pd.DataFrame) -> pd.DataFrame:
    """多表 JOIN：把学历代码翻译成文字，顺带看有没有对不上的代码。"""
    sql = """
    WITH edu_map(education, edu_name) AS (
        VALUES (1, 'graduate_school'), (2, 'university'), (3, 'high_school'),
               (4, 'others'), (0, 'UNDOCUMENTED_0')
    )
    SELECT c.education,
           COALESCE(m.edu_name, 'NO_MATCH') AS edu_name,
           COUNT(*) AS n,
           ROUND(AVG(c.default_payment_next_month), 4) AS bad_rate
    FROM credit c
    LEFT JOIN edu_map m ON c.education = m.education
    GROUP BY c.education, m.edu_name
    ORDER BY c.education
    """
    return _connect(df).execute(sql).fetchdf()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    )

    from credit.data_loader import clean, load_raw

    df = clean(load_raw())

    for name, fn in [
        ("q1 ROW_NUMBER", q1_row_number_top_n),
        ("q2 LAG(环比)", q2_lag_mom),
        ("q3 NTILE(五分位)", q3_rank_quantile),
        ("q4 CTE(聚合筛选)", q4_cte_aggregate),
        ("q5 JOIN(学历映射)", q5_multi_join),
    ]:
        print(f"\n===== {name} =====")
        print(fn(df).to_string(index=False))
