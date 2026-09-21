"""EDA：出图 + 落盘 WOE 建模宽表。"""

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无窗口后端：脚本里出图不弹窗，直接存文件
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

from credit.data_loader import TARGET, clean, load_raw, split_xy
from credit.features import build_iv_table, fit_woe, select_by_iv, woe_transform

logger = logging.getLogger(__name__)

FIG_DIR = Path("reports/figures")
PROCESSED = Path("data/processed")


def fig_target_distribution(y: pd.Series, save_dir: Path = FIG_DIR) -> Path:
    """图 1：目标变量分布——看类别是否平衡。"""
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / "01_target_distribution.png"

    counts = y.value_counts().sort_index()
    ax = sns.barplot(x=counts.index.astype(str), y=counts.values, color="#4C72B0")
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v}\n({v / len(y):.2%})", ha="center", va="bottom", fontsize=10)
    ax.set_title("Target distribution: default next month")
    ax.set_xlabel("0 = good, 1 = default")
    ax.set_ylabel("count")
    ax.set_ylim(0, counts.max() * 1.2)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    return path


def fig_missing(df: pd.DataFrame, save_dir: Path = FIG_DIR) -> Path:
    """图 2：缺失情况——如果总数是 0，那本身就是一条结论。"""
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / "02_missing.png"

    miss = df.isna().sum().sort_values(ascending=False)
    ax = sns.barplot(x=miss.values, y=miss.index, color="#DD8452")
    ax.set_title(f"Missing values per column (total = {int(miss.sum())})")
    ax.set_xlabel("n_missing")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    return path


def fig_quintile_vs_default(df: pd.DataFrame, y: pd.Series, save_dir: Path = FIG_DIR) -> Path:
    """图 3：额度五分位 vs 违约率——看单调性。"""
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / "03_limit_quintile_vs_default.png"

    tmp = pd.DataFrame({"limit_bal": df["limit_bal"], "y": y.to_numpy()})
    tmp["quintile"] = pd.qcut(tmp["limit_bal"], 5, labels=False) + 1
    rate = tmp.groupby("quintile")["y"].agg(["mean", "size"]).reset_index()

    ax = sns.barplot(x="quintile", y="mean", data=rate, color="#55A868")
    for i, row in rate.iterrows():
        ax.text(i, row["mean"], f"{row['mean']:.2%}", ha="center", va="bottom", fontsize=10)
    ax.set_title("Default rate by limit_bal quintile (monotonic?)")
    ax.set_xlabel("quintile of limit_bal (1 = lowest)")
    ax.set_ylabel("default rate")
    ax.set_ylim(0, rate["mean"].max() * 1.25)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    return path


def fig_education_vs_default(df: pd.DataFrame, y: pd.Series, save_dir: Path = FIG_DIR) -> Path:
    """图 4：学历档 vs 违约率——把未声明类别值 0/5/6 可视化出来。"""
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / "04_education_vs_default.png"

    tmp = pd.DataFrame({"education": df["education"], "y": y.to_numpy()})
    rate = tmp.groupby("education")["y"].agg(["mean", "size"]).reset_index()

    ax = sns.barplot(x="education", y="mean", data=rate, color="#C44E52")
    for i, row in rate.iterrows():
        ax.text(i, row["mean"], f"n={int(row['size'])}", ha="center", va="bottom", fontsize=9)
    ax.set_title("Default rate by EDUCATION (codes 0/5/6 undocumented)")
    ax.set_xlabel("EDUCATION code")
    ax.set_ylabel("default rate")
    ax.set_ylim(0, rate["mean"].max() * 1.3)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    return path


def build_woe_matrix(df: pd.DataFrame, y: pd.Series, keep_thr: float = 0.02) -> tuple:
    """先切分，再只在训练集上拟合 WOE，最后 transform 两个集合。"""
    X = df.drop(columns=[TARGET])
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    iv_table = build_iv_table(X_tr, y_tr)
    spec = fit_woe(X_tr, y_tr, iv_table, keep_thr=keep_thr)

    train_m = woe_transform(X_tr, spec)
    test_m = woe_transform(X_te, spec)
    train_m["y"] = y_tr.to_numpy()
    test_m["y"] = y_te.to_numpy()
    logger.info("WOE 宽表：train %s / test %s，入选特征 %d 个", train_m.shape, test_m.shape, len(spec))
    return train_m, test_m, iv_table


def run_all() -> None:
    df = clean(load_raw())
    _, y = split_xy(df)

    logger.info("图 1/4：目标分布")
    fig_target_distribution(y)
    logger.info("图 2/4：缺失情况")
    fig_missing(df)
    logger.info("图 3/4：额度五分位 vs 违约率")
    fig_quintile_vs_default(df, y)
    logger.info("图 4/4：学历档 vs 违约率")
    fig_education_vs_default(df, y)

    train_m, test_m, iv_table = build_woe_matrix(df, y)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    train_m.to_csv(PROCESSED / "features.csv", index=False)
    test_m.to_csv(PROCESSED / "features_test.csv", index=False)
    iv_table.to_csv("reports/iv_table.csv", index=False)
    logger.info("已落盘：%s / %s", PROCESSED / "features.csv", "reports/iv_table.csv")
    logger.info("入选特征（IV >= 0.02）：%s", select_by_iv(iv_table, thr=0.02))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    )
    run_all()
