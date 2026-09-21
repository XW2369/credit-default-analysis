"""信贷违约数据加载与清洗。"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

RAW_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "raw"
    / "default of credit card clients.xls"
)
TARGET = "default_payment_next_month"  # 注意：这是 clean() 之后的列名


def load_raw(path: Path | str = RAW_PATH) -> pd.DataFrame:
    """读取 UCI 原始 .xls。

    header=1：文件第 0 行是空的，真正的列名在第 1 行。
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"找不到数据文件：{path.resolve()}")

    df = pd.read_excel(path, header=1)
    logger.info("读入 %s：%s 行 x %s 列", path.name, *df.shape)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """结构上确定该做的清洗。

    只做两件确定的事：
      - 丢掉 ID 列：它是行号，没有预测价值，却容易被模型当成信号
      - 列名统一成小写 + 下划线："default payment next month" 这种带空格的列名没法用点号访问

    未声明的类别值（比如 EDUCATION=0）不在这里改——那要单独报告，不是随手抹掉。
    """
    out = df.copy()
    if "ID" in out.columns:
        out = out.drop(columns=["ID"])
    out.columns = [c.strip().lower().replace(" ", "_") for c in out.columns]
    return out


def basic_profile(df: pd.DataFrame) -> pd.DataFrame:
    """基础画像：每列的类型 / 缺失数 / 缺失率 / 唯一值数。"""
    return pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "n_missing": df.isna().sum(),
            "pct_missing": (df.isna().mean() * 100).round(2),
            "n_unique": df.nunique(),
        }
    )


def split_xy(df: pd.DataFrame, target: str = TARGET):
    """拆出特征矩阵 X 和目标 y。"""
    y = df[target].astype(int)
    X = df.drop(columns=[target])
    return X, y


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    raw = load_raw()
    logger.info("原始形状：%s", raw.shape)

    df = clean(raw)
    logger.info("清洗后形状：%s", df.shape)
    logger.info("清洗后列名：%s", list(df.columns))

    profile = basic_profile(df)
    logger.info("缺失总数：%d", int(profile["n_missing"].sum()))

    X, y = split_xy(df)
    logger.info("X %s / y %s，违约率 %.4f", X.shape, y.shape, y.mean())
