"""
美股动量排序工具 v2
依赖: pip install yfinance pandas tabulate
"""

import sys
import argparse
import calendar
import warnings
from datetime import datetime, date

warnings.filterwarnings("ignore")


# ─── 股票池获取 ───────────────────────────────────────────────────────────────

def fetch_sp500() -> "pd.DataFrame":
    import pandas as pd
    print("  获取 S&P 500 成分股（Wikipedia）...")
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    df = pd.read_html(url)[0]
    df = df[["Symbol", "Security", "GICS Sector"]].copy()
    df.columns = ["Ticker", "Name", "Sector"]
    df["Ticker"] = df["Ticker"].str.replace(".", "-", regex=False)
    df["Index"] = "S&P 500"
    return df.dropna(subset=["Ticker"])


def fetch_nasdaq100() -> "pd.DataFrame":
    import pandas as pd
    print("  获取 NASDAQ-100 成分股（Wikipedia）...")
    url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    tables = pd.read_html(url)
    for t in tables:
        cols = list(t.columns)
        low = [str(c).lower() for c in cols]
        ticker_col = next((cols[i] for i, c in enumerate(low) if c in ("ticker", "symbol")), None)
        if ticker_col is None:
            continue
        name_col   = next((cols[i] for i, c in enumerate(low) if any(k in c for k in ("company", "security", "name"))), None)
        sector_col = next((cols[i] for i, c in enumerate(low) if "sector" in c), None)
        result = pd.DataFrame()
        result["Ticker"] = t[ticker_col].astype(str).str.strip().str.replace(".", "-", regex=False)
        result["Name"]   = t[name_col].astype(str).str.strip() if name_col else ""
        result["Sector"] = t[sector_col].astype(str).str.strip() if sector_col else "N/A"
        result["Index"]  = "NASDAQ-100"
        result = result[result["Ticker"].str.match(r"^[A-Z]")]
        if len(result) > 50:
            return result.reset_index(drop=True)
    raise ValueError("无法解析 NASDAQ-100 Wikipedia 页面，请检查网络")


def build_universe(universe: str) -> "pd.DataFrame":
    import pandas as pd
    if universe == "sp500":
        return fetch_sp500()
    if universe == "nasdaq100":
        return fetch_nasdaq100()
    # both: 合并，重复股票标注双重指数
    sp = fetch_sp500()
    nq = fetch_nasdaq100()
    combined = pd.concat([sp, nq], ignore_index=True)
    grp = (
        combined.groupby("Ticker")
        .agg(Name=("Name", "first"), Sector=("Sector", "first"),
             Index=("Index", lambda x: " & ".join(sorted(set(x)))))
        .reset_index()
    )
    return grp


# ─── 价格下载 ─────────────────────────────────────────────────────────────────

def load_prices(tickers: list, start: str, end: str) -> "pd.DataFrame":
    import yfinance as yf
    import pandas as pd

    print(f"  下载 {len(tickers)} 只股票行情 ({start} → {end}) ...")
    BATCH = 100
    frames = []
    for i in range(0, len(tickers), BATCH):
        batch = tickers[i : i + BATCH]
        raw = yf.download(batch, start=start, end=end, auto_adjust=True, progress=False)
        close = raw["Close"] if "Close" in raw.columns else raw
        if isinstance(close, type(None)) or (hasattr(close, "empty") and close.empty):
            continue
        if hasattr(close, "squeeze") and close.ndim == 1:
            close = close.to_frame(name=batch[0])
        frames.append(close)
        if len(tickers) > BATCH:
            print(f"    已处理 {min(i+BATCH, len(tickers))}/{len(tickers)} ...", end="\r")

    if not frames:
        raise SystemExit("行情下载失败，请检查网络")

    prices = pd.concat(frames, axis=1)
    prices = prices.loc[:, ~prices.columns.duplicated()]
    return prices.dropna(how="all")


# ─── 动量计算 ─────────────────────────────────────────────────────────────────

def calc_momentum(prices: "pd.DataFrame", meta: "pd.DataFrame") -> "pd.DataFrame":
    import numpy as np

    meta_idx = meta.set_index("Ticker") if not meta.empty else meta
    monthly = prices.resample("ME").last()
    results = []

    for ticker in prices.columns:
        s = monthly[ticker].dropna()
        if len(s) < 3:
            continue

        monthly_ret = s.pct_change().dropna()
        total_ret   = s.iloc[-1] / s.iloc[0] - 1
        mom_mean    = monthly_ret.mean()
        mom_std     = monthly_ret.std()
        sharpe      = mom_mean / mom_std if mom_std > 1e-9 else float("nan")

        cum     = (1 + monthly_ret).cumprod()
        max_dd  = ((cum - cum.cummax()) / cum.cummax()).min()

        info = meta_idx.loc[ticker] if (not meta_idx.empty and ticker in meta_idx.index) else {}
        results.append({
            "Ticker":    ticker,
            "名称":      str(info.get("Name", "")),
            "板块":      str(info.get("Sector", "N/A")),
            "指数":      str(info.get("Index", "")),
            "当前价":    round(prices[ticker].dropna().iloc[-1], 2),
            "总收益%":   round(total_ret * 100, 2),
            "月均动量%": round(mom_mean * 100, 3),
            "动量Sharpe":round(sharpe, 3) if not (sharpe != sharpe) else "N/A",
            "最大回撤%": round(max_dd * 100, 2),
        })

    import pandas as pd
    df = pd.DataFrame(results).sort_values("月均动量%", ascending=False).reset_index(drop=True)
    df.insert(0, "排名", range(1, len(df) + 1))
    return df


# ─── 日期解析 ─────────────────────────────────────────────────────────────────

def resolve_dates(start_arg: str | None, end_arg: str | None, months: int) -> tuple[str, str]:
    """
    规则：
      --start A --end B   → 使用 A～B
      --start A           → A ～ 今天
      --end B             → B往前推 months 个月 ～ B
      (无)                → 今天往前推 months 个月 ～ 今天
    """
    def month_last_day(ym: str) -> date:
        dt = datetime.strptime(ym, "%Y-%m")
        last = calendar.monthrange(dt.year, dt.month)[1]
        return dt.replace(day=last).date()

    def month_first_day(ym: str) -> date:
        return datetime.strptime(ym, "%Y-%m").date()

    def subtract_months(d: date, n: int) -> date:
        m = d.month - n
        y = d.year
        while m <= 0:
            m += 12
            y -= 1
        return d.replace(year=y, month=m, day=1)

    end_dt = month_last_day(end_arg) if end_arg else date.today()

    if start_arg:
        start_dt = month_first_day(start_arg)
    else:
        start_dt = subtract_months(end_dt, months)

    return start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d")


# ─── 主程序 ───────────────────────────────────────────────────────────────────

GICS_SECTORS = [
    "Communication Services", "Consumer Discretionary", "Consumer Staples",
    "Energy", "Financials", "Health Care", "Industrials",
    "Information Technology", "Materials", "Real Estate", "Utilities",
]


def main():
    parser = argparse.ArgumentParser(
        description="美股动量排序 | S&P500 / NASDAQ-100 | 自定义日期区间",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
示例:
  python momentum_rank.py                                      # S&P500, 近6个月
  python momentum_rank.py --universe nasdaq100                 # NASDAQ-100
  python momentum_rank.py --universe both --index nasdaq100    # 两池取交集筛NASDAQ
  python momentum_rank.py --sector "Information Technology"    # 仅科技板块
  python momentum_rank.py --start 2024-06 --end 2024-11        # 指定区间
  python momentum_rank.py --end 2024-11 --months 6             # 2024-06 ~ 2024-11
  python momentum_rank.py --list-sectors                       # 查看全部板块名
""",
    )

    parser.add_argument(
        "--universe", choices=["sp500", "nasdaq100", "both"], default="sp500",
        metavar="UNIVERSE",
        help="股票池: sp500 | nasdaq100 | both  (默认: sp500，约500只)",
    )
    parser.add_argument(
        "--index", choices=["sp500", "nasdaq100"], default=None,
        metavar="INDEX",
        help="在 both 模式下进一步筛选所属指数: sp500 | nasdaq100",
    )
    parser.add_argument(
        "--sector", type=str, default=None, metavar="SECTOR",
        help='按 GICS 板块筛选，支持部分匹配，例: --sector "Technology"',
    )
    parser.add_argument(
        "--list-sectors", action="store_true",
        help="列出全部 GICS 板块名后退出",
    )

    dg = parser.add_argument_group("日期区间（二选一方式）")
    dg.add_argument("--months", type=int, default=6, metavar="N",
        help="往前回溯N个月（默认6），以 --end 为锚点；与 --start 同时指定时忽略")
    dg.add_argument("--start", type=str, default=None, metavar="YYYY-MM",
        help="自定义起始年月，例: 2024-06")
    dg.add_argument("--end", type=str, default=None, metavar="YYYY-MM",
        help="自定义结束年月（默认当月），例: 2024-11")

    parser.add_argument("--tickers", nargs="+", default=None,
        help="完全自定义股票列表，跳过 universe 拉取")
    parser.add_argument("--top", type=int, default=30,
        help="展示前N名（0=全部，默认30）")
    parser.add_argument("--csv", type=str, default=None,
        help="导出完整结果到 CSV，例: --csv result.csv")

    args = parser.parse_args()

    if args.list_sectors:
        print("\n可用 GICS 板块:")
        for s in GICS_SECTORS:
            print(f"  {s}")
        print()
        return

    try:
        import pandas as pd
    except ImportError:
        sys.exit("缺少依赖: pip install yfinance pandas tabulate")

    start_str, end_str = resolve_dates(args.start, args.end, args.months)

    print(f"\n{'='*62}")
    print(f"  美股动量排序  |  {start_str}  →  {end_str}")
    print(f"{'='*62}")

    # 构建股票池
    if args.tickers:
        meta = pd.DataFrame({
            "Ticker": args.tickers, "Name": "", "Sector": "N/A", "Index": "Custom"
        })
        print(f"  自定义股票池: {len(args.tickers)} 只")
    else:
        meta = build_universe(args.universe)

        # 按指数筛选（仅 both 模式有意义）
        if args.index:
            label = "S&P 500" if args.index == "sp500" else "NASDAQ-100"
            meta = meta[meta["Index"].str.contains(label, case=False)]

        # 按板块筛选（部分匹配，大小写不敏感）
        if args.sector:
            mask = meta["Sector"].str.lower().str.contains(args.sector.lower())
            meta = meta[mask]
            if meta.empty:
                sys.exit(
                    f"未找到匹配板块 '{args.sector}'，"
                    f"运行 --list-sectors 查看可用板块名"
                )

        filter_info = []
        if args.index:
            filter_info.append(f"指数={args.index.upper()}")
        if args.sector:
            filter_info.append(f"板块≈{args.sector}")
        filter_str = "  |  ".join(filter_info) if filter_info else "无筛选"
        print(f"  股票池: {args.universe}  |  {filter_str}  |  共 {len(meta)} 只")

    tickers = meta["Ticker"].tolist()
    prices  = load_prices(tickers, start_str, end_str)
    df      = calc_momentum(prices, meta)

    top_n = args.top if args.top > 0 else len(df)
    print(f"\n  有效计算: {len(df)} 只  |  展示前 {top_n} 名（月均动量降序）\n")

    try:
        from tabulate import tabulate
        print(tabulate(df.head(top_n), headers="keys", tablefmt="rounded_outline", showindex=False))
    except ImportError:
        print(df.head(top_n).to_string(index=False))

    # 摘要
    t5 = df.head(5)["Ticker"].tolist()
    b5 = df.tail(5)["Ticker"].tolist()
    print(f"\n  动量 Top5 : {' | '.join(t5)}")
    print(f"  动量 Bot5 : {' | '.join(b5)}")

    # Top20 板块分布
    if len(df) >= 10 and df["板块"].nunique() > 1:
        top20_sec = df.head(20)["板块"].value_counts()
        print(f"\n  Top20 板块分布:")
        for sec, cnt in top20_sec.items():
            bar = "█" * cnt
            print(f"    {sec:<35} {bar} {cnt}")
    print()

    if args.csv:
        df.to_csv(args.csv, index=False, encoding="utf-8-sig")
        print(f"  已导出: {args.csv}\n")


if __name__ == "__main__":
    main()
