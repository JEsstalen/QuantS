"""
美股多策略排序工具 · 本地 Web 服务
策略: 动量 / 动量+质量 / 低波动 / Piotroski F-Score
依赖: pip install flask yfinance pandas
启动: python momentum_server.py  →  http://localhost:5001
"""

import json, calendar, warnings, threading, os, math
from datetime import datetime, date
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore")
from flask import Flask, Response, request, jsonify, send_from_directory

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder=BASE_DIR)
app.config["JSON_AS_ASCII"] = False

# ─── SSE ──────────────────────────────────────────────────────────────────────

_sse_queues: list = []
_sse_lock = threading.Lock()

def _broadcast(event: str, data: object):
    def clean(obj):
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [clean(v) for v in obj]
        return obj
    msg = f"event: {event}\ndata: {json.dumps(clean(data), ensure_ascii=False)}\n\n"
    with _sse_lock:
        dead = [q for q in _sse_queues if not _try_put(q, msg)]
        for q in dead:
            _sse_queues.remove(q)

def _try_put(q, msg):
    try: q.put_nowait(msg); return True
    except: return False

def log(text: str, pct: int = -1):
    _broadcast("log", {"msg": text, "pct": pct})

# ─── Wikipedia 股票池 ──────────────────────────────────────────────────────────

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}

def _read_html_wiki(url):
    import requests, pandas as pd
    from io import StringIO
    resp = requests.get(url, headers=_HEADERS, timeout=20)
    resp.raise_for_status()
    return pd.read_html(StringIO(resp.text))

def fetch_sp500():
    import pandas as pd
    log("获取 S&P 500 成分股…", 5)
    df = _read_html_wiki("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
    df = df[["Symbol", "Security", "GICS Sector"]].copy()
    df.columns = ["Ticker", "Name", "Sector"]
    df["Ticker"] = df["Ticker"].str.replace(".", "-", regex=False)
    df["Index"] = "S&P 500"
    return df.dropna(subset=["Ticker"])

def fetch_nasdaq100():
    import pandas as pd
    log("获取 NASDAQ-100 成分股…", 5)
    for t in _read_html_wiki("https://en.wikipedia.org/wiki/Nasdaq-100"):
        cols = list(t.columns)
        low  = [str(c).lower() for c in cols]
        tc = next((cols[i] for i,c in enumerate(low) if c in ("ticker","symbol")), None)
        if not tc: continue
        nc = next((cols[i] for i,c in enumerate(low) if any(k in c for k in ("company","security","name"))), None)
        sc = next((cols[i] for i,c in enumerate(low) if "sector" in c), None)
        r = pd.DataFrame({
            "Ticker": t[tc].astype(str).str.strip().str.replace(".","-",regex=False),
            "Name":   t[nc].astype(str).str.strip() if nc else "",
            "Sector": t[sc].astype(str).str.strip() if sc else "N/A",
            "Index":  "NASDAQ-100",
        })
        r = r[r["Ticker"].str.match(r"^[A-Z]")]
        if len(r) > 50: return r.reset_index(drop=True)
    raise ValueError("无法解析 NASDAQ-100 页面")

def build_universe(universe, sector_filter):
    import pandas as pd
    if universe == "sp500":       meta = fetch_sp500()
    elif universe == "nasdaq100": meta = fetch_nasdaq100()
    else:
        meta = pd.concat([fetch_sp500(), fetch_nasdaq100()], ignore_index=True)
        meta = (meta.groupby("Ticker")
                .agg(Name=("Name","first"), Sector=("Sector","first"),
                     Index=("Index", lambda x: " & ".join(sorted(set(x)))))
                .reset_index())
    if sector_filter and sector_filter != "all":
        meta = meta[meta["Sector"].str.lower().str.contains(sector_filter.lower())]
    log(f"股票池就绪：{len(meta)} 只", 10)
    return meta.reset_index(drop=True)

# ─── 价格下载 ──────────────────────────────────────────────────────────────────

def load_prices(tickers, start, end):
    import pandas as pd, yfinance as yf
    BATCH, frames, total = 100, [], len(tickers)
    for i in range(0, total, BATCH):
        batch = tickers[i:i+BATCH]
        log(f"下载行情 {min(i+BATCH,total)}/{total}…", 10+int(min(i+BATCH,total)/total*45))
        raw = yf.download(batch, start=start, end=end, auto_adjust=True, progress=False)
        close = raw["Close"] if "Close" in raw.columns else raw
        if close is None or (hasattr(close,"empty") and close.empty): continue
        if hasattr(close,"ndim") and close.ndim==1: close = close.to_frame(name=batch[0])
        frames.append(close)
    if not frames: raise RuntimeError("行情下载失败，请检查网络")
    prices = pd.concat(frames, axis=1)
    prices = prices.loc[:, ~prices.columns.duplicated()]
    return prices.dropna(how="all")

# ─── 全局基本面缓存 ────────────────────────────────────────────────────────────
# key: ticker  value: dict of raw fields
# 缓存有效期内（同一服务进程）切换策略不重复拉取

import time as _time
_info_cache: dict = {}          # ticker -> (ts, data)
_pio_cache:  dict = {}          # ticker -> (ts, score, detail)
_CACHE_TTL = 3600               # 1小时过期

def _cache_get_info(ticker):
    entry = _info_cache.get(ticker)
    if entry and _time.time() - entry[0] < _CACHE_TTL:
        return entry[1]
    return None

def _cache_set_info(ticker, data):
    _info_cache[ticker] = (_time.time(), data)

def _cache_get_pio(ticker):
    entry = _pio_cache.get(ticker)
    if entry and _time.time() - entry[0] < _CACHE_TTL:
        return entry[1], entry[2]
    return None, None

def _cache_set_pio(ticker, score, detail):
    _pio_cache[ticker] = (_time.time(), score, detail)

# ─── 工具函数 ──────────────────────────────────────────────────────────────────

def _safe(v):
    if v is None: return None
    if isinstance(v, float) and (v != v or abs(v) == float("inf")): return None
    return v

def _monthly_stats(prices, ticker):
    """返回月度动量相关指标，失败返回 None"""
    import numpy as np
    s = prices[ticker].resample("ME").last().dropna()
    if len(s) < 3: return None
    mr = s.pct_change().dropna()
    mom_mean = mr.mean()
    mom_std  = mr.std()
    sharpe   = mom_mean/mom_std if mom_std>1e-9 else float("nan")
    cum = (1+mr).cumprod()
    max_dd = ((cum-cum.cummax())/cum.cummax()).min()
    return {
        "price":     round(float(prices[ticker].dropna().iloc[-1]), 2),
        "total_ret": round(float(s.iloc[-1]/s.iloc[0]-1)*100, 2),
        "mom_mean":  round(float(mom_mean)*100, 3),
        "sharpe":    round(float(sharpe),3) if not (sharpe!=sharpe) else None,
        "max_dd":    round(float(max_dd)*100, 2),
    }

def _meta_info(meta_idx, ticker):
    info = meta_idx.loc[ticker] if ticker in meta_idx.index else {}
    return {
        "name":   str(info.get("Name","")),
        "sector": str(info.get("Sector","N/A")),
        "index":  str(info.get("Index","")),
    }

# ─── 策略 1：纯动量 ────────────────────────────────────────────────────────────

def calc_momentum(prices, meta, start_str, end_str):
    import pandas as pd
    meta_idx = meta.set_index("Ticker")
    price_stats, total = {}, len(prices.columns)
    for idx, ticker in enumerate(prices.columns):
        if idx%50==0: log(f"计算动量 {idx}/{total}…", 58+int(idx/total*22))
        ps = _monthly_stats(prices, ticker)
        if ps: price_stats[ticker] = ps

    valid = list(price_stats.keys())
    FIELDS = ["shortPercentOfFloat","institutionsPercentHeld"]
    fundamentals = _parallel_info(valid, FIELDS, "基本面", 81, 96, workers=20)

    results = []
    for t in valid:
        fs = fundamentals.get(t, {})
        sf = round(fs["shortPercentOfFloat"]*100,1)    if fs.get("shortPercentOfFloat")    is not None else None
        ih = round(fs["institutionsPercentHeld"]*100,1) if fs.get("institutionsPercentHeld") is not None else None
        results.append({**_meta_info(meta_idx, t), "ticker": t, **price_stats[t], "short_float": sf, "inst_hold": ih})

    df = pd.DataFrame(results).sort_values("mom_mean", ascending=False).reset_index(drop=True)
    df.insert(0,"rank",range(1,len(df)+1))
    log("计算完成", 100)
    return df

# ─── 策略 2：动量 + 质量 ───────────────────────────────────────────────────────

def _fetch_info(ticker, fields):
    """拉取 yfinance .info，命中缓存直接返回；单次请求超时 8 秒"""
    cached = _cache_get_info(ticker)
    if cached is not None:
        return ticker, {k: cached.get(k) for k in fields}
    try:
        import yfinance as yf
        import threading as _th
        result = [None]
        def _do():
            try: result[0] = yf.Ticker(ticker).info
            except: pass
        t = _th.Thread(target=_do, daemon=True)
        t.start()
        t.join(timeout=8)
        if result[0] is None:
            return ticker, {k: None for k in fields}
        data = {k: _safe(result[0].get(k)) for k in (
            "returnOnEquity","profitMargins","priceToSalesTrailing12Months","beta",
            "trailingPE","forwardPE","priceToBook","enterpriseToEbitda","marketCap",
            "revenueGrowth","earningsGrowth","debtToEquity","dividendYield",
            "fiftyTwoWeekHigh","fiftyTwoWeekLow","targetMeanPrice","recommendationKey",
            "shortName","sector","industry",
            "shortPercentOfFloat","institutionsPercentHeld",
        )}
        _cache_set_info(ticker, data)
        return ticker, {k: data.get(k) for k in fields}
    except:
        return ticker, {k: None for k in fields}

def _parallel_info(tickers, fields, log_prefix, pct_start, pct_end, workers=20):
    """并发拉取一批 ticker 的 info 字段，实时推进度，返回 dict[ticker->dict]"""
    results = {}
    done = [0]
    total = len(tickers)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_fetch_info, t, fields): t for t in tickers}
        for f in as_completed(futs):
            done[0] += 1
            pct = pct_start + int(done[0] / total * (pct_end - pct_start))
            log(f"{log_prefix} {done[0]}/{total}…", pct)
            try:
                t, d = f.result()
                results[t] = d
            except:
                pass
    return results

def calc_momentum_quality(prices, meta, start_str, end_str):
    import pandas as pd, numpy as np
    meta_idx = meta.set_index("Ticker")

    price_stats = {}
    total = len(prices.columns)
    for idx, t in enumerate(prices.columns):
        if idx % 50 == 0: log(f"计算动量 {idx}/{total}…", 58 + int(idx/total*12))
        ps = _monthly_stats(prices, t)
        if ps: price_stats[t] = ps

    valid = list(price_stats.keys())
    FIELDS = ["returnOnEquity", "profitMargins", "shortPercentOfFloat", "institutionsPercentHeld"]
    fundamentals = _parallel_info(valid, FIELDS, "基本面", 71, 95, workers=20)

    rows = []
    for t in valid:
        fs = fundamentals.get(t, {})
        roe = round(fs["returnOnEquity"]*100,1) if fs.get("returnOnEquity") is not None else None
        pm  = round(fs["profitMargins"]*100,1)  if fs.get("profitMargins")  is not None else None
        sf  = round(fs["shortPercentOfFloat"]*100,1)    if fs.get("shortPercentOfFloat")    is not None else None
        ih  = round(fs["institutionsPercentHeld"]*100,1) if fs.get("institutionsPercentHeld") is not None else None
        rows.append({**_meta_info(meta_idx, t), "ticker": t,
                     **price_stats[t], "roe": roe, "profit_margin": pm, "short_float": sf, "inst_hold": ih})

    df = pd.DataFrame(rows)
    def norm(col):
        s = df[col].dropna(); mn,mx = s.min(),s.max()
        if mx==mn: return pd.Series(0.5, index=df.index)
        return (df[col].fillna(mn)-mn)/(mx-mn)
    df["composite"] = (0.5*norm("mom_mean") + 0.3*norm("roe") + 0.2*norm("profit_margin")).round(3)
    df = df.sort_values("composite", ascending=False).reset_index(drop=True)
    df.insert(0,"rank",range(1,len(df)+1))
    log("计算完成", 100)
    return df

# ─── 策略 3：低波动率 ──────────────────────────────────────────────────────────

def calc_low_vol(prices, meta, start_str, end_str):
    import pandas as pd, numpy as np, yfinance as yf
    meta_idx = meta.set_index("Ticker")

    log("下载 SPY 基准…", 58)
    spy_raw = yf.download("SPY", start=start_str, end=end_str, auto_adjust=True, progress=False)
    spy_ret = spy_raw["Close"].pct_change().dropna() if not spy_raw.empty else pd.Series(dtype=float)

    results, total = [], len(prices.columns)
    for idx, ticker in enumerate(prices.columns):
        if idx%50==0: log(f"计算波动率 {idx}/{total}…", 60+int(idx/total*35))
        ps = _monthly_stats(prices, ticker)
        if ps is None: continue

        dr = prices[ticker].pct_change().dropna()
        ann_vol = round(float(dr.std()*np.sqrt(252)*100), 2)

        beta = None
        if not spy_ret.empty:
            aln = pd.concat([dr, spy_ret], axis=1).dropna()
            aln.columns = ["s","m"]
            if len(aln) >= 20:
                varm = aln["m"].var()
                if varm > 1e-12:
                    beta = round(float(aln.cov().iloc[0,1]/varm), 3)

        results.append({**_meta_info(meta_idx, ticker), "ticker": ticker,
                        **ps, "beta": beta, "ann_vol": ann_vol})

    df = pd.DataFrame(results)
    df = df.sort_values(["beta","ann_vol"], ascending=[True,True], na_position="last").reset_index(drop=True)
    df.insert(0,"rank",range(1,len(df)+1))
    log("计算完成", 100)
    return df

# ─── 策略 4：Piotroski F-Score ─────────────────────────────────────────────────

def _piotroski_one(ticker):
    import yfinance as yf, numpy as np

    # 命中缓存直接返回
    cached_score, cached_detail = _cache_get_pio(ticker)
    if cached_score is not None or cached_detail is not None:
        return ticker, cached_score, cached_detail or {}

    def get(df, keys, col=0):
        for k in (keys if isinstance(keys,list) else [keys]):
            if k in df.index:
                v = df.loc[k].iloc[col] if col < df.shape[1] else np.nan
                if v is not None and not (isinstance(v,float) and np.isnan(v)):
                    return float(v)
        return np.nan

    def sdiv(a, b):
        if np.isnan(a) or np.isnan(b) or b==0: return np.nan
        return a/b

    def sig(cond):
        try: return 1 if bool(cond) else 0
        except: return 0

    try:
        t = yf.Ticker(ticker)
        # 三张表并发拉取，每张最多等 10 秒
        with ThreadPoolExecutor(max_workers=3) as ex:
            fi = ex.submit(lambda: t.income_stmt)
            fb = ex.submit(lambda: t.balance_sheet)
            fc = ex.submit(lambda: t.cashflow)
            inc = fi.result(timeout=10)
            bal = fb.result(timeout=10)
            cf  = fc.result(timeout=10)

        if any(x is None or (hasattr(x,"empty") and x.empty) for x in [inc,bal,cf]):
            _cache_set_pio(ticker, None, {})
            return ticker, None, {}
        if inc.shape[1]<2 or bal.shape[1]<2:
            _cache_set_pio(ticker, None, {})
            return ticker, None, {}

        ni0  = get(inc, ["Net Income","NetIncome"], 0)
        ni1  = get(inc, ["Net Income","NetIncome"], 1)
        ta0  = get(bal, ["Total Assets","TotalAssets"], 0)
        ta1  = get(bal, ["Total Assets","TotalAssets"], 1)
        ocf0 = get(cf,  ["Operating Cash Flow","OperatingCashFlow"], 0)
        ltd0 = get(bal, ["Long Term Debt","LongTermDebt"], 0) if any(k in bal.index for k in ["Long Term Debt","LongTermDebt"]) else 0.0
        ltd1 = get(bal, ["Long Term Debt","LongTermDebt"], 1) if any(k in bal.index for k in ["Long Term Debt","LongTermDebt"]) else 0.0
        ca0  = get(bal, ["Current Assets","CurrentAssets"], 0)
        ca1  = get(bal, ["Current Assets","CurrentAssets"], 1)
        cl0  = get(bal, ["Current Liabilities","CurrentLiabilities"], 0)
        cl1  = get(bal, ["Current Liabilities","CurrentLiabilities"], 1)
        sh0  = get(bal, ["Ordinary Shares Number","Share Issued","CommonStock"], 0)
        sh1  = get(bal, ["Ordinary Shares Number","Share Issued","CommonStock"], 1)
        gp0  = get(inc, ["Gross Profit","GrossProfit"], 0)
        gp1  = get(inc, ["Gross Profit","GrossProfit"], 1)
        rv0  = get(inc, ["Total Revenue","Revenue"], 0)
        rv1  = get(inc, ["Total Revenue","Revenue"], 1)

        roa0=sdiv(ni0,ta0); roa1=sdiv(ni1,ta1); ocfa=sdiv(ocf0,ta0)
        cr0=sdiv(ca0,cl0);  cr1=sdiv(ca1,cl1)
        lev0=sdiv(ltd0,ta0);lev1=sdiv(ltd1,ta1)
        gm0=sdiv(gp0,rv0);  gm1=sdiv(gp1,rv1)
        at0=sdiv(rv0,ta0);  at1=sdiv(rv1,ta1)

        f1=sig(roa0>0); f2=sig(ocfa>0)
        f3=sig(not np.isnan(roa0) and not np.isnan(roa1) and roa0>roa1)
        f4=sig(not np.isnan(ocfa) and not np.isnan(roa0) and ocfa>roa0)
        f5=sig(not np.isnan(lev0) and not np.isnan(lev1) and lev0<lev1)
        f6=sig(not np.isnan(cr0)  and not np.isnan(cr1)  and cr0>cr1)
        f7=sig(not np.isnan(sh0)  and not np.isnan(sh1)  and sh0<=sh1)
        f8=sig(not np.isnan(gm0)  and not np.isnan(gm1)  and gm0>gm1)
        f9=sig(not np.isnan(at0)  and not np.isnan(at1)  and at0>at1)

        score  = f1+f2+f3+f4+f5+f6+f7+f8+f9
        detail = {
            "f_profit":    f1+f2+f3+f4,
            "f_leverage":  f5+f6+f7,
            "f_efficiency":f8+f9,
            "roa":         round(roa0*100,2) if not np.isnan(roa0) else None,
            "gross_margin":round(gm0*100,1)  if not np.isnan(gm0)  else None,
        }
        _cache_set_pio(ticker, score, detail)
        return ticker, score, detail
    except Exception:
        _cache_set_pio(ticker, None, {})
        return ticker, None, {}

def calc_piotroski(prices, meta, start_str, end_str):
    import pandas as pd
    meta_idx = meta.set_index("Ticker")
    tickers  = list(prices.columns)
    total    = len(tickers)
    log(f"获取财务报表（{total} 只，并行中…）", 58)

    pio_data = {}
    done = [0]
    # workers=12：每只内部再起3线程，实际并发约36，不过雅虎有限流
    with ThreadPoolExecutor(max_workers=12) as ex:
        futs = {ex.submit(_piotroski_one, t): t for t in tickers}
        for f in as_completed(futs):
            done[0] += 1
            pct = 58 + int(done[0] / total * 38)
            log(f"F-Score {done[0]}/{total}…", pct)
            try:
                t, score, detail = f.result()
                pio_data[t] = (score, detail)
            except: pass

    results = []
    for ticker in tickers:
        s = prices[ticker].dropna()
        if s.empty: continue
        score, detail = pio_data.get(ticker, (None, {}))
        monthly = prices[ticker].resample("ME").last().dropna()
        total_ret = round(float(monthly.iloc[-1]/monthly.iloc[0]-1)*100,2) if len(monthly)>=2 else None
        results.append({
            **_meta_info(meta_idx, ticker),
            "ticker":       ticker,
            "price":        round(float(s.iloc[-1]),2),
            "fscore":       score,
            "f_profit":     detail.get("f_profit"),
            "f_leverage":   detail.get("f_leverage"),
            "f_efficiency": detail.get("f_efficiency"),
            "roa":          detail.get("roa"),
            "gross_margin": detail.get("gross_margin"),
            "total_ret":    total_ret,
        })

    df = pd.DataFrame(results)
    df = df.sort_values("fscore", ascending=False, na_position="last").reset_index(drop=True)
    df.insert(0,"rank",range(1,len(df)+1))
    log("计算完成", 100)
    return df

# ─── 日期解析 ──────────────────────────────────────────────────────────────────

def resolve_dates(start_ym, end_ym, months):
    def mlast(ym):
        dt = datetime.strptime(ym,"%Y-%m")
        return dt.replace(day=calendar.monthrange(dt.year,dt.month)[1]).date()
    def mfirst(ym): return datetime.strptime(ym,"%Y-%m").date()
    def subm(d,n):
        m,y = d.month-n, d.year
        while m<=0: m+=12; y-=1
        return d.replace(year=y,month=m,day=1)
    end_dt   = mlast(end_ym)   if end_ym   else date.today()
    start_dt = mfirst(start_ym) if start_ym else subm(end_dt, months)
    return start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d")

# ─── 策略 5：双动量（Dual Momentum）──────────────────────────────────────────
# 第一层：绝对动量 — 收益率 < 无风险利率（3M T-Bill）的股票排除
# 第二层：剩余股票按相对动量排序
# 无风险利率用 ^IRX（13周国库券年化收益率）近似

def calc_dual_momentum(prices, meta, start_str, end_str):
    import pandas as pd, numpy as np, yfinance as yf

    meta_idx = meta.set_index("Ticker")

    # 拉取无风险利率（^IRX = 13-week T-bill annualized %）
    log("获取无风险利率（^IRX）…", 58)
    try:
        irx = yf.download("^IRX", start=start_str, end=end_str,
                          auto_adjust=True, progress=False)["Close"].dropna()
        # IRX 是年化百分比，换算到区间总收益率
        days = (pd.Timestamp(end_str) - pd.Timestamp(start_str)).days or 1
        rf_total = float(irx.mean()) / 100 * days / 365
    except Exception:
        rf_total = 0.02  # 无法拉取时用 2% 兜底

    log(f"无风险利率（区间累计）= {rf_total*100:.2f}%，计算双动量…", 62)

    results, total = [], len(prices.columns)
    for idx, ticker in enumerate(prices.columns):
        if idx % 30 == 0:
            log(f"计算双动量 {idx}/{total}…", 62 + int(idx / total * 33))
        ps = _monthly_stats(prices, ticker)
        if ps is None:
            continue

        # 绝对动量过滤：期间总收益 > 无风险利率才保留
        abs_pass = (ps["total_ret"] / 100) > rf_total
        results.append({
            **_meta_info(meta_idx, ticker),
            "ticker":    ticker,
            **ps,
            "rf_hurdle": round(rf_total * 100, 2),
            "abs_pass":  bool(abs_pass),
        })

    df = pd.DataFrame(results)
    # 绝对动量未通过的排到最后，通过的内部按相对动量排序
    df["_sort"] = df["mom_mean"].where(df["abs_pass"], other=-9999)
    df = df.sort_values("_sort", ascending=False).drop(columns=["_sort"]).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    log("计算完成", 100)
    return df


# ─── 策略 6：多因子 Z-Score 合成 ──────────────────────────────────────────────
# 动量(35%) + 质量-ROE(25%) + 质量-利润率(15%) + 低波动-Beta(15%) + 价值-PS(10%)
# 每个因子先标准化为 Z-Score，再加权合成，最终按综合 Z 降序排列

# ─── 策略 6：多因子 Z-Score 合成 ──────────────────────────────────────────────

def calc_multifactor(prices, meta, start_str, end_str):
    import pandas as pd, numpy as np

    meta_idx = meta.set_index("Ticker")

    price_stats = {}
    total = len(prices.columns)
    for idx, t in enumerate(prices.columns):
        if idx % 30 == 0:
            log(f"计算价格因子 {idx}/{total}…", 58 + int(idx/total*12))
        ps = _monthly_stats(prices, t)
        if not ps: continue
        dr = prices[t].pct_change().dropna()
        price_stats[t] = {**ps, "ann_vol": round(float(dr.std()*np.sqrt(252)*100), 2)}

    valid  = list(price_stats.keys())
    FIELDS = ["returnOnEquity","profitMargins","priceToSalesTrailing12Months","beta","shortPercentOfFloat","institutionsPercentHeld"]
    fundamentals = _parallel_info(valid, FIELDS, "基本面", 71, 95, workers=20)

    rows = []
    for t in valid:
        fs = fundamentals.get(t, {})
        rows.append({
            **_meta_info(meta_idx, t), "ticker": t,
            **price_stats[t],
            "roe":           round(fs["returnOnEquity"]*100, 2)           if fs.get("returnOnEquity")                is not None else None,
            "profit_margin": round(fs["profitMargins"]*100, 2)            if fs.get("profitMargins")                 is not None else None,
            "ps":            round(fs["priceToSalesTrailing12Months"], 2) if fs.get("priceToSalesTrailing12Months")  is not None else None,
            "beta":          round(fs["beta"], 3)                         if fs.get("beta")                          is not None else None,
            "short_float":   round(fs["shortPercentOfFloat"]*100, 1)      if fs.get("shortPercentOfFloat")           is not None else None,
            "inst_hold":     round(fs["institutionsPercentHeld"]*100, 1)  if fs.get("institutionsPercentHeld")       is not None else None,
        })

    df = pd.DataFrame(rows)

    def zscore(col, invert=False):
        s = df[col].dropna()
        if len(s) < 2: return pd.Series(0.0, index=df.index)
        mu, sd = s.mean(), s.std()
        if sd < 1e-9: return pd.Series(0.0, index=df.index)
        z = (df[col] - mu) / sd
        return (-z if invert else z).fillna(0)

    df["mf_score"] = (
        0.35 * zscore("mom_mean") +
        0.25 * zscore("roe") +
        0.15 * zscore("profit_margin") +
        0.15 * zscore("beta", invert=True) +
        0.10 * zscore("ps",   invert=True)
    ).round(3)

    df = df.sort_values("mf_score", ascending=False).reset_index(drop=True)
    df.insert(0,"rank",range(1,len(df)+1))
    log("计算完成", 100)
    return df


# ─── 市场温度计 ───────────────────────────────────────────────────────────────

def _safe_series(s):
    import numpy as np
    return [None if (v is None or (isinstance(v, float) and np.isnan(v))) else round(float(v), 4)
            for v in s]

def _percentile_of(series, value):
    import numpy as np
    if value is None: return None
    arr = [v for v in series if v is not None]
    if not arr: return None
    return round(float(np.sum(np.array(arr) <= value) / len(arr) * 100), 1)


_market_cache = {"ts": 0, "data": None}
_MARKET_TTL = 3600  # 1小时缓存

# ─── 每个指标独立计算函数，供流式推送 ────────────────────────────────────────────
# mk() 返回的 dict 额外包含 trend20: 20日变化方向 ("up"/"down"/"flat")

def _trend20(series):
    """最新值 vs 20交易日前，返回 up/down/flat"""
    import numpy as np
    s = series.dropna()
    if len(s) < 21: return "flat"
    chg = float(s.iloc[-1]) - float(s.iloc[-21])
    thr = abs(float(s.iloc[-21])) * 0.02  # 2% 阈值判断 flat
    if chg > thr:  return "up"
    if chg < -thr: return "down"
    return "flat"

def _mk_with_trend(ctx_mk, val, chg, hist2y, hist60d, series_for_trend, rnd=2, dates60d=None):
    out = ctx_mk(val, chg, hist2y, hist60d, rnd=rnd, dates60d=dates60d)
    out["trend20"] = _trend20(series_for_trend)
    return out

# ── 战术层 ────────────────────────────────────────────────────────────────────

def _ind_vix(ctx):
    s = ctx["dl1"]("^VIX")
    s60d = s[s.index >= ctx["start_60d"]]
    chg = float(s.iloc[-1] - s.iloc[-2]) if len(s) > 1 else None
    dates = [str(d.date()) for d in s60d.index]
    return _mk_with_trend(ctx["mk"], s.iloc[-1], chg, s, s60d, s, dates60d=dates)

def _ind_put_call(ctx):
    import re as _re, requests as req
    pc_r = req.get("https://www.cboe.com/us/options/market_statistics/daily/",
                   headers=_HEADERS, timeout=10)
    pc_matches = dict(_re.findall(
        r'\{\\"name\\":\\"([^"\\\\]+PUT/CALL[^"\\\\]*)\\"[^}]*\\"value\\":\\"([^"\\\\]+)\\"',
        pc_r.text, _re.IGNORECASE))
    total_pc  = _safe(float(pc_matches["TOTAL PUT/CALL RATIO"]))  if "TOTAL PUT/CALL RATIO"  in pc_matches else None
    equity_pc = _safe(float(pc_matches["EQUITY PUT/CALL RATIO"])) if "EQUITY PUT/CALL RATIO" in pc_matches else None
    index_pc  = _safe(float(pc_matches["INDEX PUT/CALL RATIO"]))  if "INDEX PUT/CALL RATIO"  in pc_matches else None
    val = equity_pc or total_pc
    hist_ref = [0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00,1.10,1.20]
    return {
        "value": round(val, 2) if val else None, "change": None, "trend20": "flat",
        "percentile": _percentile_of(hist_ref, val) if val else None,
        "history": [], "dates": [],
        "note_extra": f"Total {total_pc}  ·  Equity {equity_pc}  ·  Index {index_pc}",
    }

def _ind_hy_spread(ctx):
    import numpy as np
    s2y, s60d, dates = ctx["fred"]("BAMLH0A0HYM2")
    val_bps = float(s2y.iloc[-1]) * 100
    chg_bps = float(s2y.iloc[-1] - s2y.iloc[-2]) * 100 if len(s2y) > 1 else None
    h2_bps  = [v*100 for v in s2y.tolist()]
    h60_bps = [v*100 for v in s60d.tolist()]
    s_bps = s2y * 100
    return {
        "value": round(val_bps, 0), "change": round(chg_bps, 0) if chg_bps else None,
        "trend20": _trend20(s_bps),
        "percentile": _percentile_of(h2_bps, val_bps),
        "history": [round(v, 0) for v in h60_bps],
        "dates": dates,
    }

def _ind_ad_ratio(ctx):
    import pandas as pd
    rsp = ctx["dl1"]("RSP"); spy = ctx["dl1"]("SPY")
    idx = rsp.index.intersection(spy.index)
    ratio = (rsp[idx] / spy[idx]).dropna()
    ma5 = ratio.rolling(5).mean().dropna()
    s60d = ma5[ma5.index >= ctx["start_60d"]]
    chg = float(ma5.iloc[-1] - ma5.iloc[-2]) if len(ma5) > 1 else None
    return _mk_with_trend(ctx["mk"], ma5.iloc[-1], chg, ma5, s60d, ma5, rnd=4)

def _ind_pct200(ctx):
    import pandas as pd
    scores = {}
    for t in ["SPY","QQQ","IWM","DIA","MDY"]:
        try:
            s = ctx["dl1"](t)
            ma200 = s.rolling(200).mean()
            above = (s > ma200).astype(float).rolling(20).mean().dropna()
            scores[t] = above
        except: pass
    if not scores: return {}
    combined = pd.concat(scores.values(), axis=1).mean(axis=1).dropna() * 100
    s60d = combined[combined.index >= ctx["start_60d"]]
    chg = float(combined.iloc[-1] - combined.iloc[-2]) if len(combined) > 1 else None
    return _mk_with_trend(ctx["mk"], combined.iloc[-1], chg, combined, s60d, combined, rnd=1)

# ── 战略层 ────────────────────────────────────────────────────────────────────

def _ind_real_rate(ctx):
    """实际利率：FRED DFII10（10年期 TIPS 收益率，%）"""
    s2y, s60d, dates = ctx["fred"]("DFII10")
    chg = float(s2y.iloc[-1] - s2y.iloc[-2]) if len(s2y) > 1 else None
    return _mk_with_trend(ctx["mk"], s2y.iloc[-1], chg, s2y, s60d, s2y, rnd=2, dates60d=dates)

def _ind_cape(ctx):
    import pandas as pd
    df = pd.read_excel("http://www.econ.yale.edu/~shiller/data/ie_data.xls",
                       sheet_name="Data", skiprows=7, engine="xlrd")
    cape = pd.to_numeric(df["CAPE"], errors="coerce").dropna().tail(300)
    val = float(cape.iloc[-1])
    return {
        "value": round(val, 1), "change": None,
        "trend20": _trend20(cape),
        "percentile": _percentile_of(cape.tolist(), val),
        "history": [round(v,1) for v in cape.iloc[-24:].tolist()],
    }

def _ind_yield_spread(ctx):
    s2y, s60d, dates = ctx["fred"]("T10Y2Y")
    chg = float(s2y.iloc[-1] - s2y.iloc[-2]) if len(s2y) > 1 else None
    return _mk_with_trend(ctx["mk"], s2y.iloc[-1], chg, s2y, s60d, s2y, rnd=2, dates60d=dates)

def _ind_dxy(ctx):
    """美元指数 DX-Y.NYB"""
    import pandas as pd
    s = ctx["dl1"]("DX-Y.NYB")
    s60d = s[s.index >= ctx["start_60d"]]
    chg = float(s.iloc[-1] - s.iloc[-2]) if len(s) > 1 else None
    dates = [str(d.date()) for d in s60d.index]
    return _mk_with_trend(ctx["mk"], s.iloc[-1], chg, s, s60d, s, rnd=1, dates60d=dates)

def _ind_margin_debt(ctx):
    s2y, s60d, dates = ctx["fred"]("BOGMBBM")
    val = float(s2y.iloc[-1]) * 10
    chg = float(s2y.iloc[-1] - s2y.iloc[-2]) * 10 if len(s2y) > 1 else None
    h2  = s2y * 10; h60 = s60d * 10
    return {
        "value": round(val, 0), "change": round(chg, 0) if chg else None,
        "trend20": _trend20(h2),
        "percentile": _percentile_of(h2.tolist(), val),
        "history": [round(v, 0) for v in h60.tolist()],
        "dates": dates,
    }

def _ind_gold(ctx):
    """黄金期货 GC=F"""
    s = ctx["dl1"]("GC=F")
    s60d = s[s.index >= ctx["start_60d"]]
    chg = float(s.iloc[-1] - s.iloc[-2]) if len(s) > 1 else None
    dates = [str(d.date()) for d in s60d.index]
    return _mk_with_trend(ctx["mk"], s.iloc[-1], chg, s, s60d, s, rnd=0, dates60d=dates)

def _ind_oil(ctx):
    """WTI 原油期货 CL=F"""
    s = ctx["dl1"]("CL=F")
    s60d = s[s.index >= ctx["start_60d"]]
    chg = float(s.iloc[-1] - s.iloc[-2]) if len(s) > 1 else None
    dates = [str(d.date()) for d in s60d.index]
    return _mk_with_trend(ctx["mk"], s.iloc[-1], chg, s, s60d, s, rnd=1, dates60d=dates)

# key / fn / weight / layer("tactical"/"strategic")
_MARKET_INDICATORS = [
    ("vix",          _ind_vix,          3.0, "tactical"),
    ("put_call",     _ind_put_call,      2.5, "tactical"),
    ("hy_spread",    _ind_hy_spread,     3.0, "tactical"),
    ("ad_ratio",     _ind_ad_ratio,      2.0, "tactical"),
    ("pct200",       _ind_pct200,        2.0, "tactical"),
    ("gold",         _ind_gold,          0.0, "tactical"),   # 0 weight = 不计入温度
    ("oil",          _ind_oil,           0.0, "tactical"),
    ("real_rate",    _ind_real_rate,     3.0, "strategic"),
    ("cape",         _ind_cape,          2.5, "strategic"),
    ("yield_spread", _ind_yield_spread,  2.0, "strategic"),
    ("dxy",          _ind_dxy,           2.0, "strategic"),
    ("margin_debt",  _ind_margin_debt,   1.5, "strategic"),
]

def _build_ctx():
    import pandas as pd, numpy as np, yfinance as yf
    today = pd.Timestamp.today()
    start_2y  = (today - pd.DateOffset(years=2)).strftime("%Y-%m-%d")
    start_60d = (today - pd.DateOffset(days=60)).strftime("%Y-%m-%d")

    def dl1(ticker):
        raw = yf.download(ticker, start=start_2y, auto_adjust=True, progress=False)
        c = raw["Close"] if "Close" in raw.columns else raw
        if hasattr(c,"ndim") and c.ndim==2: c=c.iloc[:,0]
        return c.dropna()

    def mk(val, chg, hist2y, hist60d, rnd=2, dates60d=None):
        v = round(float(val), rnd) if val is not None else None
        h2  = [round(float(x),rnd) for x in hist2y  if x is not None and not np.isnan(float(x))]
        h60 = [round(float(x),rnd) for x in hist60d if x is not None and not np.isnan(float(x))]
        c = round(float(chg), rnd) if chg is not None else None
        out = {"value":v,"change":c,"percentile":_percentile_of(h2,v),"history":h60}
        if dates60d is not None: out["dates"] = list(dates60d)
        return out

    def fred(series_id):
        import pandas as pd
        df = pd.read_csv(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}")
        df.columns = ["date","val"]
        df["val"] = pd.to_numeric(df["val"], errors="coerce")
        df = df.dropna(subset=["val"]).sort_values("date")
        df["date"] = pd.to_datetime(df["date"])
        s2y  = df[df["date"] >= pd.Timestamp(start_2y)]
        s60d = df[df["date"] >= pd.Timestamp(start_60d)]
        dates = [str(d.date()) for d in s60d["date"]]
        return s2y["val"], s60d["val"], dates

    return {"dl1":dl1,"mk":mk,"fred":fred,"start_2y":start_2y,"start_60d":start_60d}

def _fetch_market_data():
    ctx = _build_ctx()
    result = {}
    for key, fn, _weight, _proxy in _MARKET_INDICATORS:
        try:
            result[key] = fn(ctx)
        except:
            result[key] = {}
    return result

@app.route("/api/market")
def api_market():
    import time
    now = time.time()
    if now - _market_cache["ts"] < _MARKET_TTL and _market_cache["data"]:
        return jsonify(_market_cache["data"])
    try:
        data = _fetch_market_data()
        _market_cache["ts"] = now
        _market_cache["data"] = data
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/market/stream")
def api_market_stream():
    """逐指标推送 SSE，每算完一个立即发出；同时写入缓存"""
    import time, queue as _q
    force = request.args.get("force","0") == "1"
    now = time.time()
    # 缓存命中直接一次性推全量
    if not force and now - _market_cache["ts"] < _MARKET_TTL and _market_cache["data"]:
        cached = _market_cache["data"]
        def _stream_cached():
            for key, _fn, _w, _p in _MARKET_INDICATORS:
                d = cached.get(key, {})
                msg = json.dumps({"key": key, "data": d}, ensure_ascii=False)
                yield f"event: indicator\ndata: {msg}\n\n"
            yield f"event: done\ndata: {{\"ts\":{int(_market_cache['ts'])}}}\n\n"
        return Response(_stream_cached(), mimetype="text/event-stream",
                        headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

    result_acc = {}
    result_lock = threading.Lock()

    def _stream_live():
        ctx = _build_ctx()
        for key, fn, _w, _p in _MARKET_INDICATORS:
            try:
                d = fn(ctx)
            except Exception as e:
                d = {"error": str(e)}
            with result_lock:
                result_acc[key] = d
            # clean NaN before sending
            def _clean(obj):
                if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)): return None
                if isinstance(obj, dict):  return {k: _clean(v) for k,v in obj.items()}
                if isinstance(obj, list):  return [_clean(v) for v in obj]
                return obj
            msg = json.dumps({"key": key, "data": _clean(d)}, ensure_ascii=False)
            yield f"event: indicator\ndata: {msg}\n\n"
        # write cache
        ts = time.time()
        _market_cache["ts"] = ts
        _market_cache["data"] = dict(result_acc)
        yield f"event: done\ndata: {{\"ts\":{int(ts)}}}\n\n"

    return Response(_stream_live(), mimetype="text/event-stream",
                    headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})


# ─── 板块热力图 ────────────────────────────────────────────────────────────────
# 11个SPDR ETF，拉60日日涨跌，返回 {sector, dates[], returns[]} 列表

_SECTOR_ETFS = [
    ("XLK",  "信息技术"),
    ("XLV",  "医疗健康"),
    ("XLF",  "金融"),
    ("XLY",  "非必需消费"),
    ("XLP",  "必需消费"),
    ("XLE",  "能源"),
    ("XLI",  "工业"),
    ("XLU",  "公用事业"),
    ("XLRE", "房地产"),
    ("XLB",  "材料"),
    ("XLC",  "通信服务"),
]

# ─── 细分概念 ETF（24个热门主题，按市场关注度排序）────────────────────────────
_THEME_ETFS = [
    ("SMH",  "半导体"),
    ("SOXX", "半导体(iShares)"),
    ("ARKK", "颠覆创新(ARK)"),
    ("QQQM", "纳斯达克100"),
    ("BOTZ", "机器人AI"),
    ("ROBO", "机器人自动化"),
    ("HACK", "网络安全"),
    ("CIBR", "网络安全(First Trust)"),
    ("CLOU", "云计算"),
    ("FINX", "金融科技"),
    ("IBB",  "生物科技"),
    ("XBI",  "生物科技(SPDR)"),
    ("ITA",  "国防航天"),
    ("XAR",  "航空航天"),
    ("ICLN", "清洁能源"),
    ("TAN",  "太阳能"),
    ("LIT",  "锂电池"),
    ("URA",  "铀矿核能"),
    ("KWEB", "中概互联"),
    ("FXI",  "中国大盘"),
    ("EWJ",  "日本股市"),
    ("INDA", "印度"),
    ("GDX",  "黄金矿"),
    ("KRE",  "区域银行"),
    ("IBIT", "比特币ETF"),
    ("JETS", "航空"),
    ("GAMR", "游戏电竞"),
    ("CARS", "电动车"),
]

_sector_heatmap_cache = {"ts": 0, "data": None}
_theme_heatmap_cache  = {"ts": 0, "data": None}
_SECTOR_TTL = 3600

def _phase_label(returns):
    """根据近5日/近10日/近20日加速度判断热度阶段"""
    import numpy as np
    if len(returns) < 21: return None, None
    arr = [r for r in returns if r is not None]
    if len(arr) < 21: return None, None
    last5  = sum(returns[-5:])
    prev5  = sum(returns[-10:-5])
    last10 = sum(returns[-10:])
    last20 = sum(returns[-20:])
    # 加速度: 近5日 - 前5日
    accel = round(last5 - prev5, 2)
    # 阶段判定
    if last5 > 0 and last5 > prev5 and last20 > 0:
        phase = "accel_up"      # 加速上涨
    elif last5 > 0 and prev5 < 0:
        phase = "reversal_up"   # 反转向上
    elif last5 < 0 and last20 > 0:
        phase = "topping"       # 见顶回落
    elif last5 < 0 and last5 < prev5 and last20 < 0:
        phase = "accel_dn"      # 加速下跌
    elif last5 > 0 and prev5 > 0 and last5 < prev5:
        phase = "cooling"       # 涨势减速
    elif last5 < 0 and prev5 < 0 and last5 > prev5:
        phase = "stabilizing"   # 跌势减缓
    else:
        phase = "neutral"
    return phase, accel

def _compute_heatmap(etf_list):
    """通用热力计算：返回 {sectors:[...], dates:[...]}"""
    import yfinance as yf, pandas as pd, numpy as np
    today = pd.Timestamp.today()
    start = (today - pd.DateOffset(days=90)).strftime("%Y-%m-%d")
    tickers = [e[0] for e in etf_list]
    raw = yf.download(tickers, start=start, auto_adjust=True, progress=False)
    close = raw["Close"] if "Close" in raw.columns else raw
    if hasattr(close,"ndim") and close.ndim==1: close=close.to_frame(name=tickers[0])
    close = close.dropna(how="all")

    ret = close.pct_change().dropna(how="all") * 100
    ret = ret.tail(60)
    dates = [str(d.date()) for d in ret.index]

    result = []
    for ticker, name in etf_list:
        if ticker not in ret.columns: continue
        s = ret[ticker]
        vals = [round(float(v),2) if not np.isnan(v) else None for v in s]
        c5  = round(float((close[ticker].iloc[-1]/close[ticker].iloc[-6 ]-1)*100),2) if len(close)>=6  else None
        c20 = round(float((close[ticker].iloc[-1]/close[ticker].iloc[-21]-1)*100),2) if len(close)>=21 else None
        c60 = round(float((close[ticker].iloc[-1]/close[ticker].iloc[-61]-1)*100),2) if len(close)>=61 else None
        phase, accel = _phase_label(vals)
        result.append({
            "ticker": ticker, "name": name,
            "dates": dates, "returns": vals,
            "cum5": c5, "cum20": c20, "cum60": c60,
            "phase": phase, "accel": accel,
        })

    # 按近5日收益降序，让热度高的排在前面
    result.sort(key=lambda x: x.get("cum5") or -999, reverse=True)
    return {"sectors": result, "dates": dates}

@app.route("/api/sectors/heatmap")
def api_sector_heatmap():
    import time
    kind = request.args.get("kind", "sectors")
    cache = _theme_heatmap_cache if kind == "themes" else _sector_heatmap_cache
    etf_list = _THEME_ETFS if kind == "themes" else _SECTOR_ETFS
    now = time.time()
    if now - cache["ts"] < _SECTOR_TTL and cache["data"]:
        return jsonify(cache["data"])
    try:
        data = _compute_heatmap(etf_list)
        # 计算 RRG 坐标
        data = _add_rrg_coords(data, etf_list)
        cache["ts"] = now
        cache["data"] = data
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def _add_rrg_coords(data, etf_list):
    """计算每个板块相对 SPY 的 RRG 坐标
    X 轴 (RS-Ratio):  最近14日相对 SPY 强度，100 = 持平
    Y 轴 (RS-Mom):   RS-Ratio 的 5日动量，100 = 强度无变化
    四象限：
      X>100 Y>100: 领涨 (Leading)
      X<100 Y>100: 改善 (Improving)
      X<100 Y<100: 落后 (Lagging)
      X>100 Y<100: 走弱 (Weakening)
    历史轨迹保留最近 8 周用于绘制尾巴
    """
    import yfinance as yf, pandas as pd, numpy as np
    try:
        today = pd.Timestamp.today()
        start = (today - pd.DateOffset(days=180)).strftime("%Y-%m-%d")
        tickers = [e[0] for e in etf_list] + ["SPY"]
        raw = yf.download(tickers, start=start, auto_adjust=True, progress=False)
        close = raw["Close"] if "Close" in raw.columns else raw
        if "SPY" not in close.columns:
            return data
        spy = close["SPY"].dropna()

        # 对每只 ETF：rs = (price/spy) / rolling_mean(14)
        for sec in data.get("sectors", []):
            tk = sec["ticker"]
            if tk not in close.columns: continue
            s = close[tk].dropna()
            common = s.index.intersection(spy.index)
            if len(common) < 30: continue
            rs_raw = (s.loc[common] / spy.loc[common])
            # 归一化为100基准（相对 14 日均值）
            rs_norm = (rs_raw / rs_raw.rolling(14).mean()) * 100
            rs_mom = (rs_norm / rs_norm.shift(5)) * 100   # 5日动量
            rs_norm = rs_norm.dropna()
            rs_mom  = rs_mom.dropna()
            if len(rs_norm) < 2 or len(rs_mom) < 2: continue

            # 取最近 8 个数据点作为尾巴（约 8 周）
            tail_n = 8
            xs = rs_norm.tail(tail_n).tolist()
            ys = rs_mom.tail(tail_n).tolist()
            sec["rrg_x"] = round(float(xs[-1]), 2)
            sec["rrg_y"] = round(float(ys[-1]), 2)
            sec["rrg_tail_x"] = [round(float(v), 2) for v in xs]
            sec["rrg_tail_y"] = [round(float(v), 2) for v in ys]
            # 象限
            if sec["rrg_x"] >= 100 and sec["rrg_y"] >= 100:   q = "leading"
            elif sec["rrg_x"] < 100 and sec["rrg_y"] >= 100:  q = "improving"
            elif sec["rrg_x"] < 100 and sec["rrg_y"] < 100:   q = "lagging"
            else:                                              q = "weakening"
            sec["rrg_quadrant"] = q
    except Exception as e:
        log(f"RRG 计算失败：{e}", -1)
    return data


# ─── 策略 7：52周新高动量（George & Hwang 2004, JoF）───────────────────────────
# 距离52周高点的接近度本身就是动量信号，比传统动量更稳定
# 规则：dist_52w_high < 5% → 强势；同时要求 200日均线之上 + 正向月动量

def calc_52w_high(prices, meta, start_str, end_str):
    import pandas as pd, numpy as np
    meta_idx = meta.set_index("Ticker")

    results, total = [], len(prices.columns)
    for idx, ticker in enumerate(prices.columns):
        if idx % 30 == 0: log(f"计算52W新高接近度 {idx}/{total}…", 58 + int(idx/total*35))
        s = prices[ticker].dropna()
        if len(s) < 252: continue

        last = float(s.iloc[-1])
        high_52w = float(s.tail(252).max())
        low_52w  = float(s.tail(252).min())
        # 距离52周高点（%，越接近0越强）
        dist_high = round((last/high_52w - 1)*100, 2)
        # 距52周低点（%）
        from_low  = round((last/low_52w - 1)*100, 2)
        # 200日均线
        ma200 = float(s.tail(200).mean()) if len(s) >= 200 else None
        above_ma200 = ma200 is not None and last > ma200
        # 月度动量（次级筛选）
        ps = _monthly_stats(prices, ticker)
        mom = ps["mom_mean"] if ps else None
        # 双重过滤：在200日均线之上 AND 月动量为正
        passed = bool(above_ma200 and (mom is not None and mom > 0))

        results.append({
            **_meta_info(meta_idx, ticker), "ticker": ticker,
            "price":      round(last, 2),
            "high_52w":   round(high_52w, 2),
            "low_52w":    round(low_52w, 2),
            "dist_high":  dist_high,      # 越接近0越好（负数）
            "from_low":   from_low,
            "above_ma200": above_ma200,
            "mom_mean":   ps["mom_mean"] if ps else None,
            "total_ret":  ps["total_ret"] if ps else None,
            "max_dd":     ps["max_dd"] if ps else None,
            "passed":     passed,
        })

    df = pd.DataFrame(results)
    # 排序：先按通过过滤排，通过的内部按距高点近排��dist_high 越接近0越前）
    df["_sort"] = df["dist_high"].where(df["passed"], other=-9999)
    df = df.sort_values("_sort", ascending=False).drop(columns=["_sort"]).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df)+1))
    log("计算完成", 100)
    return df

# ─── 策略 8：Connors RSI(2) 超卖反弹（Larry Connors）─────────────────────────
# 短线 mean reversion：200日均线之上（趋势过滤）+ RSI(2) 极低（极度超卖）
# 标准入场：RSI(2) < 5；放宽到 < 10 进入观察名单

def _rsi(series, n):
    import pandas as pd
    delta = series.diff()
    up = delta.clip(lower=0)
    dn = -delta.clip(upper=0)
    # Wilder's smoothing — 用 Connors 原版（简单平均）
    avg_up = up.rolling(n).mean()
    avg_dn = dn.rolling(n).mean()
    rs = avg_up / avg_dn.replace(0, 1e-12)
    return 100 - 100/(1+rs)

def calc_connors_rsi(prices, meta, start_str, end_str):
    import pandas as pd, numpy as np
    meta_idx = meta.set_index("Ticker")

    results, total = [], len(prices.columns)
    for idx, ticker in enumerate(prices.columns):
        if idx % 30 == 0: log(f"计算 Connors RSI {idx}/{total}…", 58 + int(idx/total*35))
        s = prices[ticker].dropna()
        if len(s) < 200: continue

        last = float(s.iloc[-1])
        ma200 = float(s.tail(200).mean())
        ma5   = float(s.tail(5).mean())
        above_ma200 = last > ma200

        rsi2  = _rsi(s, 2)
        rsi14 = _rsi(s, 14)
        last_rsi2  = float(rsi2.iloc[-1])  if not pd.isna(rsi2.iloc[-1])  else None
        last_rsi14 = float(rsi14.iloc[-1]) if not pd.isna(rsi14.iloc[-1]) else None

        # 多周期确认：周线 RSI(14) > 50 说明大方向仍是涨
        # 周线 = 每5个交易日取一个点（近似周收盘）
        weekly = s.iloc[::-1].iloc[::5].iloc[::-1]
        weekly_rsi14 = _rsi(weekly, 14)
        last_w_rsi = float(weekly_rsi14.iloc[-1]) if len(weekly_rsi14) and not pd.isna(weekly_rsi14.iloc[-1]) else None
        weekly_bullish = last_w_rsi is not None and last_w_rsi > 50

        # Connors 标准信号：200日均线上 + RSI(2)<5 + 周线方向向上
        signal = bool(above_ma200 and weekly_bullish and last_rsi2 is not None and last_rsi2 < 5)
        # 观察名单：信号条件放宽
        watchlist = bool(above_ma200 and last_rsi2 is not None and last_rsi2 < 10)

        from_ma200 = round((last/ma200 - 1)*100, 2)

        results.append({
            **_meta_info(meta_idx, ticker), "ticker": ticker,
            "price":         round(last, 2),
            "rsi2":          round(last_rsi2, 1)  if last_rsi2  is not None else None,
            "rsi14":         round(last_rsi14, 1) if last_rsi14 is not None else None,
            "weekly_rsi":    round(last_w_rsi, 1) if last_w_rsi is not None else None,
            "weekly_bullish":weekly_bullish,
            "ma5":           round(ma5, 2),
            "ma200":         round(ma200, 2),
            "from_ma200":    from_ma200,
            "above_ma200":   above_ma200,
            "signal":        signal,
            "watchlist":     watchlist,
        })

    df = pd.DataFrame(results)
    def _bucket(r):
        if r["signal"]:    return 2
        if r["watchlist"]: return 1
        return 0
    df["_bucket"] = df.apply(_bucket, axis=1)
    df["_rsi"]    = df["rsi2"].fillna(999)
    df = df.sort_values(["_bucket","_rsi"], ascending=[False, True]).drop(columns=["_bucket","_rsi"]).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df)+1))
    log("计算完成", 100)
    return df


STRATEGIES = {
    "momentum":         calc_momentum,
    "momentum_quality": calc_momentum_quality,
    "low_vol":          calc_low_vol,
    "piotroski":        calc_piotroski,
    "dual_momentum":    calc_dual_momentum,
    "multifactor":      calc_multifactor,
    "high52w":          calc_52w_high,
    "connors_rsi":      calc_connors_rsi,
}

# ─── 目标价 / 止损测算（前10名）─────────────────────────────────────────────
# 用 ATR(14) 倍数法 + 期限分组，参考 Welles Wilder 经典思路
# short  : 持有 2-5 天   止损 1.5R 目标 1.5R（mean reversion，看 MA20）
# mid    : 持有 1-3 月   止损 2.0R 目标 4.0R（2:1 风险回报）
# long   : 持有 6-18 月  止损 2.5R 目标 7.5R（3:1 风险回报）
_TERM_MAP = {
    "connors_rsi":      ("short", 1.5, 1.5, "2-5天",  "mean reversion → MA20"),
    "momentum":         ("mid",   2.0, 4.0, "1-3月",  "动量延续，2:1 R:R"),
    "momentum_quality": ("mid",   2.0, 4.0, "1-3月",  "质量动量，2:1 R:R"),
    "low_vol":          ("mid",   1.8, 3.0, "1-3月",  "低波动，紧止损"),
    "dual_momentum":    ("mid",   2.0, 4.0, "1-3月",  "双动量，2:1 R:R"),
    "multifactor":      ("mid",   2.0, 4.0, "1-3月",  "多因子，2:1 R:R"),
    "piotroski":        ("long",  2.5, 7.5, "6-18月", "基本面长线，3:1 R:R"),
    "high52w":          ("long",  2.5, 7.5, "6-18月", "52W动量长线，3:1 R:R"),
}

def _atr(prices, ticker, n=14):
    """ATR(n) — 用日收盘价近似（缺 H/L 数据时的简化版）"""
    import pandas as pd, numpy as np
    s = prices[ticker].dropna()
    if len(s) < n+1: return None
    # 用日收益绝对值近似 True Range（无 high/low/close 完整数据时的常见做法）
    tr = s.diff().abs()
    atr = tr.rolling(n).mean().iloc[-1]
    if pd.isna(atr) or atr <= 0: return None
    return float(atr)

# ─── 真 ATR（带 H/L/C 完整 True Range）─────────────────────────────────────
# 用 yfinance.history() 单只补拉，仅前10名用 — 不影响整体计算时间

def _atr_true(ticker, n=14, dollar_vol=False):
    """返回 (atr_true, avg_dollar_vol_20d) 或 (None, None)
    True Range = max(H-L, |H-Cprev|, |Cprev-L|)
    """
    import pandas as pd, numpy as np
    try:
        import yfinance as yf
        result = [None]
        def _do():
            try:
                # 60天足够算 14日 ATR + 20日成交额
                result[0] = yf.Ticker(ticker).history(period="60d", auto_adjust=True)
            except: pass
        import threading as _th
        th = _th.Thread(target=_do, daemon=True)
        th.start(); th.join(timeout=6)
        df = result[0]
        if df is None or df.empty or len(df) < n+1:
            return None, None
        high  = df["High"]
        low   = df["Low"]
        close = df["Close"]
        prev_close = close.shift(1)
        tr = pd.concat([(high-low).abs(),
                        (high-prev_close).abs(),
                        (low-prev_close).abs()], axis=1).max(axis=1)
        atr = float(tr.rolling(n).mean().iloc[-1])
        if pd.isna(atr) or atr <= 0:
            return None, None
        # 20日平均美元成交额
        adv = None
        if dollar_vol and "Volume" in df.columns and len(df) >= 20:
            dv = (close * df["Volume"]).tail(20)
            adv = float(dv.mean())
        return atr, adv
    except:
        return None, None

# ─── 财报日期（事件风险）─────────────────────────────────────────────────────

_earnings_cache: dict = {}     # ticker -> (ts, date_str|None)
_EARNINGS_TTL = 6 * 3600       # 6 小时缓存

def _fetch_earnings(ticker):
    """返回 (ticker, 'YYYY-MM-DD' 或 None)"""
    import time as _t
    entry = _earnings_cache.get(ticker)
    if entry and _t.time() - entry[0] < _EARNINGS_TTL:
        return ticker, entry[1]
    try:
        import yfinance as yf, pandas as pd
        result = [None]
        def _do():
            try:
                t = yf.Ticker(ticker)
                cal = t.calendar
                if isinstance(cal, dict):
                    ed = cal.get("Earnings Date")
                    if isinstance(ed, list) and ed:
                        result[0] = ed[0]
                    elif ed is not None:
                        result[0] = ed
            except: pass
        import threading as _th
        th = _th.Thread(target=_do, daemon=True)
        th.start(); th.join(timeout=6)
        date_str = None
        if result[0] is not None:
            try:
                dt = pd.to_datetime(result[0])
                date_str = dt.strftime("%Y-%m-%d")
            except: pass
        _earnings_cache[ticker] = (_t.time(), date_str)
        return ticker, date_str
    except:
        _earnings_cache[ticker] = (_t.time(), None)
        return ticker, None

def _parallel_earnings(tickers, pct_start, pct_end, workers=10):
    results = {}
    done = [0]; total = len(tickers)
    if total == 0: return results
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_fetch_earnings, t): t for t in tickers}
        for f in as_completed(futs):
            done[0] += 1
            pct = pct_start + int(done[0] / total * (pct_end - pct_start))
            log(f"财报日期 {done[0]}/{total}…", pct)
            try:
                t, d = f.result()
                results[t] = d
            except: pass
    return results

def _balance_by_sector(df, max_per_sector=2, top_n=10):
    """从排序后的 df 中挑 top_n 只，每个板块最多 max_per_sector 只
    依然按原排序优先级挑选，跳过超额板块的股票
    """
    import pandas as pd
    if df.empty: return df.head(0), []
    picked_idx = []
    sector_count = {}
    skipped = []   # 被跳过的股票（ticker, sector, reason）
    for i in df.index:
        sec = df.at[i, "sector"]
        sec_key = str(sec) if sec is not None and not pd.isna(sec) else "未知"
        cnt = sector_count.get(sec_key, 0)
        if cnt >= max_per_sector:
            skipped.append({
                "ticker": df.at[i, "ticker"],
                "sector": sec_key,
                "orig_rank": int(df.at[i, "rank"]) if "rank" in df.columns else None,
            })
            continue
        picked_idx.append(i)
        sector_count[sec_key] = cnt + 1
        if len(picked_idx) >= top_n:
            break
    balanced = df.loc[picked_idx].copy()
    return balanced, skipped

def _add_targets(df, prices, strategy, balance_sectors=False, max_per_sector=2):
    """给前10名计算目标价 / 止损 / 风险回报比 / 财报日期 / 流动性
    使用 True Range ATR（带 H/L/C 数据），并标记低流动性股票
    """
    import pandas as pd, numpy as np
    if df.empty: return df, None
    term, stop_mult, target_mult, horizon, basis = _TERM_MAP.get(
        strategy, ("mid", 2.0, 4.0, "1-3月", "2:1 R:R"))

    df["term"]    = term
    df["horizon"] = horizon
    for col in ("target","stop","risk_reward","upside_pct","downside_pct",
                "earnings_date","days_to_earnings","atr","dollar_vol_20d","low_liq"):
        df[col] = None

    # 板块均衡：从排行榜按规则筛 10 只
    skipped = []
    if balance_sectors:
        balanced, skipped = _balance_by_sector(df, max_per_sector=max_per_sector, top_n=10)
        top10_idx = list(balanced.index)
    else:
        top10_idx = list(df.head(10).index)
    top10_tickers = df.loc[top10_idx, "ticker"].tolist()

    # 并行拉前10名财报日期
    earnings_map = _parallel_earnings(top10_tickers, 96, 99, workers=10)

    today_dt = pd.Timestamp.today().normalize()
    for i in top10_idx:
        ticker = df.at[i, "ticker"]
        entry = df.at[i, "price"]
        if entry is None or pd.isna(entry): continue

        # 真 ATR + 流动性（H/L/C + Volume）
        atr_t, adv = _atr_true(ticker, n=14, dollar_vol=True)
        # fallback：拉不到 H/L 时用 close 近似
        if atr_t is None and ticker in prices.columns:
            atr_t = _atr(prices, ticker, 14)
        if atr_t is None: continue

        stop_price   = entry - stop_mult   * atr_t
        target_price = entry + target_mult * atr_t

        # Connors 短线：目标用 MA20（mean reversion）
        if strategy == "connors_rsi" and ticker in prices.columns:
            s = prices[ticker].dropna()
            if len(s) >= 20:
                ma20 = float(s.tail(20).mean())
                target_price = max(target_price, ma20)

        upside   = (target_price/entry - 1) * 100
        downside = (1 - stop_price/entry)   * 100
        rr = target_mult / stop_mult

        df.at[i, "target"]       = round(float(target_price), 2)
        df.at[i, "stop"]         = round(float(stop_price),   2)
        df.at[i, "upside_pct"]   = round(float(upside),       2)
        df.at[i, "downside_pct"] = round(float(downside),     2)
        df.at[i, "risk_reward"]  = round(float(rr),           2)
        df.at[i, "atr"]          = round(float(atr_t),        2)

        # 流动性：日均美元成交额；< $5M 视为低流动性
        if adv is not None:
            df.at[i, "dollar_vol_20d"] = round(float(adv) / 1e6, 1)  # 单位：百万美元
            df.at[i, "low_liq"]        = bool(adv < 5e6)

        # 财报日
        ed = earnings_map.get(ticker)
        if ed:
            try:
                ed_dt = pd.to_datetime(ed).normalize()
                days = int((ed_dt - today_dt).days)
                df.at[i, "earnings_date"]    = ed
                df.at[i, "days_to_earnings"] = days
            except: pass

    # 如果做了板块均衡：把均衡后的10只搬到表头，原排名仍保留在 orig_rank
    if balance_sectors:
        df["orig_rank"] = df["rank"]
        balanced_rows = df.loc[top10_idx].copy()
        rest = df.drop(top10_idx)
        df = pd.concat([balanced_rows, rest], ignore_index=False).reset_index(drop=True)
        df["rank"] = range(1, len(df) + 1)
    return df, skipped if balance_sectors else None

# ─── 前10诊断：相关性 + 板块占比 ──────────────────────────────────────────────

def _diagnose_top10(df, prices):
    """计算前10名 60日相关性、板块占比、平均相关性"""
    import pandas as pd, numpy as np
    top = df.head(10)
    if top.empty: return None
    tickers = [t for t in top["ticker"].tolist() if t in prices.columns]
    if len(tickers) < 2: return None

    # 60日相关性矩阵
    daily = prices[tickers].pct_change().dropna(how="all").tail(60)
    if len(daily) < 10: return None
    corr = daily.corr()

    matrix = []
    for i, ta in enumerate(tickers):
        row = []
        for tb in tickers:
            v = corr.loc[ta, tb]
            row.append(round(float(v), 2) if not pd.isna(v) else None)
        matrix.append(row)

    # 平均相关性（去对角线）
    n = len(tickers)
    sum_off = sum(matrix[i][j] for i in range(n) for j in range(n) if i != j and matrix[i][j] is not None)
    cnt_off = sum(1 for i in range(n) for j in range(n) if i != j and matrix[i][j] is not None)
    avg_corr = round(sum_off / cnt_off, 3) if cnt_off else None

    # 板块占比
    sectors = {}
    for sec in top["sector"].tolist():
        if not sec or pd.isna(sec): sec = "未知"
        sec = str(sec)
        sectors[sec] = sectors.get(sec, 0) + 1
    sector_bars = [{"sector": s, "count": c, "pct": round(c/len(top)*100, 0)}
                   for s, c in sorted(sectors.items(), key=lambda x: -x[1])]

    # 集中度警告
    max_sec = sector_bars[0] if sector_bars else None
    warnings_list = []
    if max_sec and max_sec["count"] >= 5:
        warnings_list.append(f"⚠ {max_sec['sector']} 占 {max_sec['count']}/10，板块过度集中")
    if avg_corr is not None and avg_corr > 0.7:
        warnings_list.append(f"⚠ 平均相关性 {avg_corr}，10只走势高度同向，分散效果弱")
    elif avg_corr is not None and avg_corr > 0.5:
        warnings_list.append(f"近60日平均相关性 {avg_corr}，分散效果一般")

    return {
        "tickers":      tickers,
        "matrix":       matrix,
        "avg_corr":     avg_corr,
        "sector_bars":  sector_bars,
        "warnings":     warnings_list,
    }

# ─── 详情接口 ─────────────────────────────────────────────────────────────────

@app.route("/api/detail/<ticker>")
def api_detail(ticker):
    import yfinance as yf
    months = int(request.args.get("months", 6))
    start, end = resolve_dates(request.args.get("start"), request.args.get("end"), months)
    t    = yf.Ticker(ticker)
    hist = t.history(start=start, end=end, auto_adjust=True)
    prices = [{"date":str(d.date()),"close":round(float(c),2),"volume":int(v)}
              for d,c,v in zip(hist.index,hist["Close"],hist["Volume"]) if c==c]
    info = {}
    try:
        raw = t.info
        info = {k: _safe(raw.get(v)) for k,v in {
            "pe":"trailingPE","forward_pe":"forwardPE","pb":"priceToBook",
            "ps":"priceToSalesTrailing12Months","ev_ebitda":"enterpriseToEbitda",
            "market_cap":"marketCap","revenue_growth":"revenueGrowth",
            "earnings_growth":"earningsGrowth","profit_margin":"profitMargins",
            "roe":"returnOnEquity","debt_equity":"debtToEquity",
            "dividend_yield":"dividendYield","beta":"beta",
            "52w_high":"fiftyTwoWeekHigh","52w_low":"fiftyTwoWeekLow",
            "analyst_target":"targetMeanPrice","recommendation":"recommendationKey",
            "short_name":"shortName","sector":"sector","industry":"industry",
            "short_float":"shortPercentOfFloat","inst_hold":"institutionsPercentHeld",
        }.items()}
    except: pass
    return jsonify({"ticker": ticker, "prices": prices, "info": info})

# ─── 主计算接口 ───────────────────────────────────────────────────────────────

_calc_lock = threading.Lock()

@app.route("/api/run", methods=["POST"])
def api_run():
    body = request.json or {}

    def worker():
        try:
            import pandas as pd
            meta = build_universe(body.get("universe","nasdaq100"), body.get("sector_filter",""))
            strategy = body.get("strategy", "momentum")
            months = int(body.get("months",6))
            # 短线/技术策略需要至少 12 个月计算 200日均线、52周高低点
            if strategy in ("high52w", "connors_rsi") and months < 12:
                months = 12
            start_str, end_str = resolve_dates(body.get("start"), body.get("end"), months)
            log(f"区间：{start_str} → {end_str}", 12)
            prices = load_prices(meta["Ticker"].tolist(), start_str, end_str)
            fn = STRATEGIES.get(strategy, calc_momentum)
            df = fn(prices, meta, start_str, end_str)
            balance_sectors = bool(body.get("balance_sectors", False))
            max_per_sector  = int(body.get("max_per_sector", 2))
            df, skipped = _add_targets(df, prices, strategy,
                                       balance_sectors=balance_sectors,
                                       max_per_sector=max_per_sector)
            diagnostics = _diagnose_top10(df, prices)
            rows = df.head(100).to_dict(orient="records")
            _broadcast("done", {"rows": rows, "start": start_str, "end": end_str,
                                "strategy": strategy, "diagnostics": diagnostics,
                                "skipped": skipped, "balance_sectors": balance_sectors})
        except Exception as e:
            _broadcast("error", {"msg": str(e)})

    if not _calc_lock.acquire(blocking=False):
        return jsonify({"error": "已有任务运行中，请稍候"}), 429

    def run_release():
        try: worker()
        finally: _calc_lock.release()

    threading.Thread(target=run_release, daemon=True).start()
    return jsonify({"ok": True})

# ─── 简版回测（walk-forward）────────────────────────────────────────────────
# 每月初按当前策略选前 N 只，等权持有一个月，到月底再换；与基准（SPY）对比
# 缓存因为重算成本高（拉历史 + 反复算策略）

_backtest_cache: dict = {}   # key -> (ts, result)
_BACKTEST_TTL = 6 * 3600

def _run_backtest(strategy_key, universe, sector_filter, lookback_months, hold_n=10, cost_bps=10, balance_sectors=False, max_per_sector=2):
    """走前向回测 — 每月初按策略选前 N 只持有一个月
    cost_bps: 每次换仓的双边交易成本（bps，含滑点）。10 bps = 0.1% per turnover
    balance_sectors: 是否每板块最多 max_per_sector 只
    """
    import pandas as pd, numpy as np, yfinance as yf
    # 不同策略需要的最小信号窗口不同
    # connors_rsi / high52w 需要 ≥200 交易日（≈10 个月），其它 6 个月够
    SIG_WIN_MONTHS = 12 if strategy_key in ("connors_rsi", "high52w") else 6
    bal_tag = "板块均衡" if balance_sectors else "原始排序"
    log(f"回测准备：{strategy_key} · 回看 {lookback_months} 月 · {bal_tag} · 成本 {cost_bps}bps", 5)
    cost_rate = cost_bps / 10000.0

    # 1. 建股票池
    meta = build_universe(universe, sector_filter)

    # 2. 一次性拉所有价格（多拉一段作信号窗口）
    end = pd.Timestamp.today()
    start = end - pd.DateOffset(months=lookback_months + SIG_WIN_MONTHS)
    start_str, end_str = start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
    log("下载历史价格…", 12)
    prices_full = load_prices(meta["Ticker"].tolist(), start_str, end_str)

    # 3. 基准 SPY
    log("下载 SPY 基准…", 35)
    spy = yf.download("SPY", start=start_str, end=end_str, auto_adjust=True, progress=False)["Close"].dropna()
    if hasattr(spy, "ndim") and spy.ndim == 2: spy = spy.iloc[:, 0]

    # 4. 找月初日期序列（每个月第一个交易日），取最近 lookback_months 个
    monthly = prices_full.resample("ME").last()
    month_ends = monthly.index[-(lookback_months+1):]   # 多取1个作起点

    # 5. 每个月底用前6个月数据跑策略 → 选前N → 持有到下月底
    fn = STRATEGIES.get(strategy_key, calc_momentum)
    equity = [1.0]              # 策略净值（起点1）
    bench  = [1.0]              # SPY 净值
    dates  = [str(month_ends[0].date())]
    wins, losses = 0, 0
    trade_returns = []
    selected_log = []           # 每月持仓记录

    for i in range(len(month_ends) - 1):
        sig_end   = month_ends[i]
        hold_end  = month_ends[i+1]
        # 信号窗口：按策略需要
        sig_start = sig_end - pd.DateOffset(months=SIG_WIN_MONTHS)
        pct_progress = 40 + int((i / max(1, len(month_ends)-2)) * 55)
        log(f"回测 {sig_end.strftime('%Y-%m')}…", pct_progress)

        # 切信号窗口数据
        prices_sig = prices_full.loc[(prices_full.index >= sig_start) & (prices_full.index <= sig_end)]
        prices_sig = prices_sig.dropna(axis=1, how="all")
        if prices_sig.empty or len(prices_sig) < 30:
            equity.append(equity[-1])
            bench.append(bench[-1])
            dates.append(str(hold_end.date()))
            continue

        # 跑策略
        try:
            df_rank = fn(prices_sig, meta, sig_start.strftime("%Y-%m-%d"), sig_end.strftime("%Y-%m-%d"))
        except Exception:
            df_rank = None
        if df_rank is None or df_rank.empty:
            equity.append(equity[-1])
            bench.append(bench[-1])
            dates.append(str(hold_end.date()))
            continue

        # 板块均衡（如开启）
        if balance_sectors and "sector" in df_rank.columns:
            balanced, _skipped = _balance_by_sector(df_rank, max_per_sector=max_per_sector, top_n=hold_n)
            top = balanced["ticker"].tolist()
        else:
            top = df_rank.head(hold_n)["ticker"].tolist()
        # 持有期收益：sig_end → hold_end，等权
        port_ret = 0.0
        cnt = 0
        for t in top:
            if t not in prices_full.columns: continue
            s = prices_full[t]
            p0 = s.asof(sig_end)
            p1 = s.asof(hold_end)
            if pd.isna(p0) or pd.isna(p1) or p0 <= 0: continue
            r = float(p1/p0 - 1)
            port_ret += r
            cnt += 1
            trade_returns.append(r)
            if r > 0: wins += 1
            else:    losses += 1
        if cnt > 0: port_ret /= cnt
        # 扣除换仓成本（每月100%换仓，每只买+卖各一次 = 双边）
        port_ret -= cost_rate * 2  # buy + sell

        # SPY 期间收益（基准也扣一次单边成本作为公平对比）
        try:
            sp0 = float(spy.asof(sig_end))
            sp1 = float(spy.asof(hold_end))
            spy_ret = sp1/sp0 - 1 if sp0 > 0 else 0
        except: spy_ret = 0

        equity.append(equity[-1] * (1 + port_ret))
        bench.append(bench[-1]  * (1 + spy_ret))
        dates.append(str(hold_end.date()))
        selected_log.append({"date": str(sig_end.date()), "tickers": top, "ret": round(port_ret*100, 2)})

    # 6. 汇总统计
    arr_eq   = np.array(equity)
    cum_ret  = (arr_eq[-1] - 1) * 100
    cum_bench = (bench[-1] - 1) * 100
    # 年化
    n_months = len(equity) - 1
    n_years = n_months / 12 if n_months > 0 else 1
    cagr = ((arr_eq[-1])**(1/n_years) - 1) * 100 if n_years > 0 else 0
    # 最大回撤
    peak = np.maximum.accumulate(arr_eq)
    dd = (arr_eq - peak) / peak
    max_dd = float(dd.min()) * 100
    # 胜率
    total_trades = wins + losses
    win_rate = wins / total_trades * 100 if total_trades else 0
    # Sharpe（月度，年化）
    monthly_rets = np.diff(arr_eq) / arr_eq[:-1]
    if len(monthly_rets) > 1 and monthly_rets.std() > 1e-9:
        sharpe = (monthly_rets.mean() / monthly_rets.std()) * np.sqrt(12)
    else:
        sharpe = 0
    # 超额
    alpha = cum_ret - cum_bench

    log("回测完成", 100)
    return {
        "strategy":   strategy_key,
        "lookback":   lookback_months,
        "hold_n":     hold_n,
        "cost_bps":   cost_bps,
        "balance_sectors": balance_sectors,
        "dates":      dates,
        "equity":     [round(float(v), 4) for v in arr_eq],
        "bench":      [round(float(v), 4) for v in bench],
        "stats": {
            "cum_ret":    round(float(cum_ret),   2),
            "bench_ret":  round(float(cum_bench), 2),
            "alpha":      round(float(alpha),     2),
            "cagr":       round(float(cagr),      2),
            "max_dd":     round(float(max_dd),    2),
            "win_rate":   round(float(win_rate),  1),
            "sharpe":     round(float(sharpe),    2),
            "trades":     int(total_trades),
            "wins":       int(wins),
            "losses":     int(losses),
        },
        "history":    selected_log[-12:],   # 最近12个月持仓
    }

_backtest_lock = threading.Lock()

@app.route("/api/backtest", methods=["POST"])
def api_backtest():
    body = request.json or {}
    strategy = body.get("strategy", "momentum")
    universe = body.get("universe", "nasdaq100")
    sector_filter = body.get("sector_filter", "")
    lookback = int(body.get("lookback_months", 24))
    hold_n   = int(body.get("hold_n", 10))
    cost_bps = int(body.get("cost_bps", 10))
    balance_sectors = bool(body.get("balance_sectors", False))
    cache_key = f"{strategy}|{universe}|{sector_filter}|{lookback}|{hold_n}|{cost_bps}|{int(balance_sectors)}"
    import time as _t
    entry = _backtest_cache.get(cache_key)
    if entry and _t.time() - entry[0] < _BACKTEST_TTL:
        return jsonify(entry[1])

    def worker():
        try:
            result = _run_backtest(strategy, universe, sector_filter, lookback, hold_n, cost_bps, balance_sectors=balance_sectors)
            _backtest_cache[cache_key] = (_t.time(), result)
            _broadcast("backtest_done", result)
        except Exception as e:
            _broadcast("error", {"msg": "回测失败：" + str(e)})

    if not _backtest_lock.acquire(blocking=False):
        return jsonify({"error": "已有回测运行中，请稍候"}), 429

    def run_release():
        try: worker()
        finally: _backtest_lock.release()

    threading.Thread(target=run_release, daemon=True).start()
    return jsonify({"ok": True})

_consensus_cache: dict = {}
_CONSENSUS_TTL = 1800   # 30 分钟

@app.route("/api/consensus", methods=["POST"])
def api_consensus():
    """跨策略叠加榜：把所有策略各自的前 N 名汇总，按出现次数排序
    NOTE: 共享 _calc_lock 避免和 /api/run 并发跑同一份股票池
    """
    body = request.json or {}
    universe = body.get("universe", "nasdaq100")
    sector_filter = body.get("sector_filter", "")
    months = int(body.get("months", 6))
    top_n  = int(body.get("top_n", 10))

    cache_key = f"{universe}|{sector_filter}|{months}|{top_n}"
    import time as _t
    entry = _consensus_cache.get(cache_key)
    if entry and _t.time() - entry[0] < _CONSENSUS_TTL:
        return jsonify(entry[1])

    if not _calc_lock.acquire(blocking=False):
        return jsonify({"error": "已有任务运行中，请稍候"}), 429

    def worker():
        try:
            import pandas as pd
            meta = build_universe(universe, sector_filter)
            # 不同策略需要的最小窗口不同，统一拉 12 个月够所有用
            m = max(months, 12)
            start_str, end_str = resolve_dates(None, None, m)
            log(f"叠加榜：拉取 {start_str} → {end_str} 价格…", 8)
            prices = load_prices(meta["Ticker"].tolist(), start_str, end_str)

            # 选要跑的策略（排除非常慢的 piotroski - 它要拉财报）
            strats = ["momentum","momentum_quality","low_vol","dual_momentum",
                      "multifactor","high52w","connors_rsi"]
            # piotroski 需要单独拉财报，太慢，叠加榜里跳过

            ticker_score = {}     # ticker -> {"count":n, "ranks":{strat:rank}, "name":..., "sector":..., "price":...}
            for si, sk in enumerate(strats):
                pct = 12 + int(si / len(strats) * 80)
                log(f"叠加榜：跑策略 {sk} ({si+1}/{len(strats)})…", pct)
                try:
                    fn = STRATEGIES.get(sk)
                    df = fn(prices, meta, start_str, end_str)
                    top = df.head(top_n)
                    for _, row in top.iterrows():
                        tk = row.get("ticker")
                        if not tk: continue
                        entry = ticker_score.setdefault(tk, {
                            "ticker": tk,
                            "name":   row.get("name", ""),
                            "sector": row.get("sector", "N/A"),
                            "price":  row.get("price"),
                            "count":  0,
                            "ranks":  {},
                        })
                        entry["count"] += 1
                        entry["ranks"][sk] = int(row.get("rank")) if row.get("rank") is not None else None
                except Exception as e:
                    log(f"叠加榜：策略 {sk} 失败：{e}", pct)
                    continue

            # 按出现次数排序，相同次数按平均排名升序
            def _avg_rank(item):
                ranks = [r for r in item["ranks"].values() if r is not None]
                return sum(ranks)/len(ranks) if ranks else 999
            results = sorted(ticker_score.values(),
                             key=lambda x: (-x["count"], _avg_rank(x)))
            for i, r in enumerate(results):
                r["overall_rank"] = i + 1
                r["avg_rank"] = round(_avg_rank(r), 1)

            payload = {
                "rows":      results[:30],
                "strategies": strats,
                "universe":  universe,
                "months":    m,
                "top_n":     top_n,
            }
            _consensus_cache[cache_key] = (_t.time(), payload)
            _broadcast("consensus_done", payload)
            log("叠加榜完成", 100)
        except Exception as e:
            _broadcast("error", {"msg": "叠加榜失败：" + str(e)})

    def run_release():
        try: worker()
        finally: _calc_lock.release()

    threading.Thread(target=run_release, daemon=True).start()
    return jsonify({"ok": True})

# ─── 多策略对比回测 ──────────────────────────────────────────────────────────

@app.route("/api/backtest/compare", methods=["POST"])
def api_backtest_compare():
    """对所有策略用同一参数跑回测，逐个完成 → 前端实时累积"""
    body = request.json or {}
    universe = body.get("universe", "nasdaq100")
    sector_filter = body.get("sector_filter", "")
    lookback = int(body.get("lookback_months", 24))
    hold_n   = int(body.get("hold_n", 10))
    cost_bps = int(body.get("cost_bps", 10))
    balance_sectors = bool(body.get("balance_sectors", False))
    # 完整 8 个策略；piotroski 单只就慢，5个月以下也别跑
    strategies = ["momentum","momentum_quality","low_vol","dual_momentum",
                  "multifactor","high52w","connors_rsi","piotroski"]

    if not _backtest_lock.acquire(blocking=False):
        return jsonify({"error": "已有回测运行中，请稍候"}), 429

    def worker():
        import time as _t
        try:
            results = {}
            for i, sk in enumerate(strategies):
                key = f"{sk}|{universe}|{sector_filter}|{lookback}|{hold_n}|{cost_bps}|{int(balance_sectors)}"
                entry = _backtest_cache.get(key)
                if entry and _t.time() - entry[0] < _BACKTEST_TTL:
                    log(f"[{i+1}/{len(strategies)}] {sk}：缓存命中", int((i+1)/len(strategies)*100))
                    res = entry[1]
                else:
                    log(f"[{i+1}/{len(strategies)}] {sk}：开始回测…", int(i/len(strategies)*100))
                    try:
                        res = _run_backtest(sk, universe, sector_filter, lookback, hold_n, cost_bps, balance_sectors=balance_sectors)
                        _backtest_cache[key] = (_t.time(), res)
                    except Exception as e:
                        log(f"[{i+1}/{len(strategies)}] {sk} 失败：{e}", -1)
                        continue
                results[sk] = {
                    "stats":  res["stats"],
                    "equity": res["equity"],
                    "dates":  res["dates"],
                }
                # 实时推送：每跑完一个，推一次
                _broadcast("compare_partial", {"strategy": sk, "data": results[sk]})

            _broadcast("compare_done", {
                "results": results,
                "bench_equity": res["bench"] if results else None,
                "lookback": lookback, "cost_bps": cost_bps,
                "balance_sectors": balance_sectors,
                "universe": universe,
            })
        except Exception as e:
            _broadcast("error", {"msg": "对比回测失败：" + str(e)})

    def run_release():
        try: worker()
        finally: _backtest_lock.release()

    threading.Thread(target=run_release, daemon=True).start()
    return jsonify({"ok": True})

# ─── 多策略对比回测 END ─────────────────────────────────────────────────────

@app.route("/api/sectors")
def api_sectors():
    import pandas as pd
    universe = request.args.get("universe","sp500")
    try:
        if universe=="sp500":       meta=fetch_sp500()
        elif universe=="nasdaq100": meta=fetch_nasdaq100()
        else: meta=pd.concat([fetch_sp500(),fetch_nasdaq100()],ignore_index=True)
        return jsonify({"sectors": sorted(meta["Sector"].dropna().unique().tolist())})
    except Exception as e:
        return jsonify({"sectors":[],"error":str(e)})

@app.route("/events")
def events():
    import queue
    q = queue.Queue(maxsize=200)
    with _sse_lock: _sse_queues.append(q)
    def stream():
        try:
            while True:
                try: yield q.get(timeout=30)
                except: yield ": ping\n\n"
        finally:
            with _sse_lock:
                if q in _sse_queues: _sse_queues.remove(q)
    return Response(stream(), mimetype="text/event-stream",
                    headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "momentum_ui.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    print(f"\n  美股多策略排序工具")
    print(f"  访问: http://localhost:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
