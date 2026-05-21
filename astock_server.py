"""
A 股多策略排序工具 · 本地 Web 服务
策略：北向资金动量 / 量价突破 / 核心动量
依赖：pip install flask akshare pandas
启动：python astock_server.py  →  http://localhost:5002
"""

import json, math, os, threading, time, warnings
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore")
from flask import Flask, Response, request, jsonify, send_from_directory

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder=BASE_DIR)
app.config["JSON_AS_ASCII"] = False

# ─── SSE ──────────────────────────────────────────────────────────────────────
_sse_queues: list = []
_sse_lock = threading.Lock()

def _broadcast(event, data):
    def clean(obj):
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)): return None
        if isinstance(obj, dict): return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, list): return [clean(v) for v in obj]
        return obj
    msg = f"event: {event}\ndata: {json.dumps(clean(data), ensure_ascii=False)}\n\n"
    with _sse_lock:
        dead = [q for q in _sse_queues if not _try_put(q, msg)]
        for q in dead:
            _sse_queues.remove(q)

def _try_put(q, msg):
    try: q.put_nowait(msg); return True
    except: return False

def log(text, pct=-1):
    _broadcast("log", {"msg": text, "pct": pct})

# ─── 缓存 ─────────────────────────────────────────────────────────────────────
_universe_cache = {"ts": 0, "data": None, "key": ""}
_prices_cache: dict = {}        # symbol -> (ts, DataFrame)
_north_cache: dict = {}         # symbol -> (ts, north_data)
_CACHE_TTL = 4 * 3600

# ─── 股票池 ──────────────────────────────────────────────────────────────────

def fetch_universe(scope="hs300_zz500"):
    """scope: hs300 / zz500 / hs300_zz500
    返回 meta DataFrame: symbol, name, index, sector, pe, pb, market_cap, turnover, qb (量比)
    """
    import akshare as ak
    import pandas as pd

    key = scope
    if _universe_cache["data"] is not None and _universe_cache["key"] == key \
            and time.time() - _universe_cache["ts"] < _CACHE_TTL:
        return _universe_cache["data"]

    frames = []
    if "hs300" in scope:
        log("获取沪深 300 成分股…", 5)
        try:
            df = ak.index_stock_cons_csindex(symbol="000300")
            df = df.rename(columns={"成分券代码": "symbol", "成分券名称": "name"})
            df["index"] = "沪深300"
            frames.append(df[["symbol", "name", "index"]])
        except Exception as e:
            log(f"沪深300 失败：{e}", -1)
    if "zz500" in scope:
        log("获取中证 500 成分股…", 8)
        try:
            df = ak.index_stock_cons_csindex(symbol="000905")
            df = df.rename(columns={"成分券代码": "symbol", "成分券名称": "name"})
            df["index"] = "中证500"
            frames.append(df[["symbol", "name", "index"]])
        except Exception as e:
            log(f"中证500 失败：{e}", -1)

    if not frames:
        raise RuntimeError("无法获取成分股")
    meta = pd.concat(frames, ignore_index=True)
    meta = meta.drop_duplicates(subset="symbol", keep="first").reset_index(drop=True)

    # 补充行业 + 估值 + 市值 + 量比 + 换手率（来自实时快照）
    log("补充行业/估值/市值/量比…", 12)
    try:
        spot = ak.stock_zh_a_spot_em()
        # 列：代码/名称/最新价/涨跌幅/...所处行业/市盈率-动态/市净率/总市值/流通市值/换手率/量比
        rename_map = {
            "代码": "_sym", "所处行业": "sector",
            "市盈率-动态": "pe", "市净率": "pb",
            "总市值": "market_cap", "流通市值": "circ_cap",
            "换手率": "turnover", "量比": "qb",
            "60日涨跌幅": "ret_60d",
        }
        spot = spot.rename(columns=rename_map)
        spot["_sym"] = spot["_sym"].astype(str)
        keep = ["_sym","sector","pe","pb","market_cap","circ_cap","turnover","qb","ret_60d"]
        keep = [c for c in keep if c in spot.columns]
        spot = spot[keep].set_index("_sym")
        meta["symbol"] = meta["symbol"].astype(str)
        meta = meta.merge(spot, how="left", left_on="symbol", right_index=True)
    except Exception as e:
        log(f"行业/估值合并失败：{e}", -1)
        for c in ("sector","pe","pb","market_cap","circ_cap","turnover","qb","ret_60d"):
            if c not in meta.columns: meta[c] = None
    meta["sector"] = meta["sector"].fillna("未知")

    _universe_cache.update(ts=time.time(), data=meta, key=key)
    log(f"股票池就绪：{len(meta)} 只", 15)
    return meta

# ─── 行情下载（按月 batch + 缓存）──────────────────────────────────────────

def _load_one_price(symbol, start_date, end_date, retries=2):
    """单只历史日 K（前复权），带重试"""
    cached = _prices_cache.get(symbol)
    if cached and time.time() - cached[0] < _CACHE_TTL:
        df = cached[1]
        if df is None: return None  # 之前拉失败已缓存
        if not df.empty and df.index[-1] >= pd.Timestamp(end_date):
            return df.loc[df.index >= pd.Timestamp(start_date)]
    import akshare as ak
    import pandas as pd
    last_err = None
    for attempt in range(retries + 1):
        try:
            df = ak.stock_zh_a_hist(symbol=symbol, period="daily",
                                    start_date=start_date.replace("-", ""),
                                    end_date=end_date.replace("-", ""), adjust="qfq")
            if df is None or df.empty:
                _prices_cache[symbol] = (time.time(), None)  # 缓存空结果避免重复尝试
                return None
            df = df.rename(columns={"日期": "date", "收盘": "close", "成交量": "volume",
                                     "成交额": "amount", "最高": "high", "最低": "low",
                                     "开盘": "open"})
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()
            _prices_cache[symbol] = (time.time(), df)
            return df
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(0.3 * (attempt + 1))   # 0.3s, 0.6s
                continue
    # 多次重试失败：短期缓存 None，避免短时间内反复重试
    _prices_cache[symbol] = (time.time(), None)
    return None

# import pandas at module level for the cache TTL check above
import pandas as pd

def load_prices(symbols, start_date, end_date, log_pct_start=15, log_pct_end=50):
    """并发下载多只股票的价格，返回 dict[symbol] -> DataFrame"""
    result = {}
    total = len(symbols)
    done = [0]
    # 18 workers：纯 I/O，akshare 服务端能扛
    with ThreadPoolExecutor(max_workers=18) as ex:
        futs = {ex.submit(_load_one_price, s, start_date, end_date): s for s in symbols}
        for f in as_completed(futs):
            done[0] += 1
            s = futs[f]
            pct = log_pct_start + int(done[0] / total * (log_pct_end - log_pct_start))
            if done[0] % 30 == 0 or done[0] == total:
                log(f"下载行情 {done[0]}/{total}…", pct)
            try:
                df = f.result()
                if df is not None and not df.empty:
                    result[s] = df
            except: pass
    return result

# ─── 月度统计 ─────────────────────────────────────────────────────────────────

def _monthly_stats(df):
    """返回月度动量相关指标"""
    import pandas as pd, numpy as np
    if df is None or df.empty or len(df) < 30: return None
    s = df["close"].resample("ME").last().dropna()
    if len(s) < 3: return None
    mr = s.pct_change().dropna()
    mom_mean = mr.mean()
    mom_std = mr.std()
    sharpe = mom_mean / mom_std if mom_std > 1e-9 else float("nan")
    cum = (1 + mr).cumprod()
    max_dd = ((cum - cum.cummax()) / cum.cummax()).min()
    return {
        "price":     round(float(df["close"].iloc[-1]), 2),
        "total_ret": round(float(s.iloc[-1] / s.iloc[0] - 1) * 100, 2),
        "mom_mean":  round(float(mom_mean) * 100, 3),
        "sharpe":    round(float(sharpe), 3) if not (sharpe != sharpe) else None,
        "max_dd":    round(float(max_dd) * 100, 2),
    }

# ─── 北向资金 ─────────────────────────────────────────────────────────────────

def fetch_north_holdings(symbols, log_pct_start=50, log_pct_end=80):
    """拉取北向资金持股比例近 60 日变化（仅对策略选定的 top 名单）"""
    import akshare as ak
    import pandas as pd
    result = {}

    def _one(symbol, retries=1):
        cached = _north_cache.get(symbol)
        if cached and time.time() - cached[0] < _CACHE_TTL:
            return symbol, cached[1]
        last_err = None
        for attempt in range(retries + 1):
            try:
                df = ak.stock_hsgt_individual_em(symbol=symbol)
                if df is None or df.empty:
                    _north_cache[symbol] = (time.time(), None)
                    return symbol, None
                col_pct = None
                for c in df.columns:
                    if "A股百分比" in c or "持股比例" in c:
                        col_pct = c; break
                col_date = None
                for c in df.columns:
                    if "日期" in c:
                        col_date = c; break
                if not col_pct or not col_date:
                    _north_cache[symbol] = (time.time(), None)
                    return symbol, None
                df = df.rename(columns={col_date: "date", col_pct: "pct"})
                df["date"] = pd.to_datetime(df["date"])
                df["pct"] = pd.to_numeric(df["pct"], errors="coerce")
                df = df.dropna(subset=["pct"]).set_index("date").sort_index()
                if df.empty:
                    _north_cache[symbol] = (time.time(), None)
                    return symbol, None
                pct_now = float(df["pct"].iloc[-1])
                chg_20d = pct_now - float(df["pct"].iloc[-21]) if len(df) >= 21 else None
                chg_60d = pct_now - float(df["pct"].iloc[-61]) if len(df) >= 61 else None
                data = {"north_pct_now": round(pct_now, 3),
                        "north_chg_20d": round(chg_20d, 3) if chg_20d is not None else None,
                        "north_chg_60d": round(chg_60d, 3) if chg_60d is not None else None}
                _north_cache[symbol] = (time.time(), data)
                return symbol, data
            except Exception as e:
                last_err = e
                if attempt < retries:
                    time.sleep(0.2 * (attempt + 1))
                    continue
        # 重试失败 → 短期缓存 None
        _north_cache[symbol] = (time.time(), None)
        return symbol, None

    total = len(symbols)
    done = [0]
    # 北向接口比价格接口慢，提到 10 并发
    with ThreadPoolExecutor(max_workers=10) as ex:
        futs = {ex.submit(_one, s): s for s in symbols}
        for f in as_completed(futs):
            done[0] += 1
            pct = log_pct_start + int(done[0] / total * (log_pct_end - log_pct_start))
            if done[0] % 30 == 0 or done[0] == total:
                log(f"北向资金 {done[0]}/{total}…", pct)
            try:
                s, d = f.result()
                if d: result[s] = d
            except: pass
    return result

# ─── 策略 1：北向资金动量 ─────────────────────────────────────────────────────

def calc_north_money(prices_map, meta, **kw):
    """北向资金持续净增加 + 月动量正
    优化：先按月动量排序选 top 150，只对这部分拉北向资金（最大耗时点）
    """
    import pandas as pd
    meta_idx = meta.set_index("symbol")
    log("策略 1：北向资金动量 - 收集价格指标…", 70)

    rows = []
    symbols = list(prices_map.keys())
    for s in symbols:
        ps = _monthly_stats(prices_map[s])
        if ps is None: continue
        info = meta_idx.loc[s] if s in meta_idx.index else {}
        rows.append({
            "symbol": s, "name": str(info.get("name", "")),
            "sector": str(info.get("sector", "未知")),
            "index": str(info.get("index", "")),
            **ps,
        })

    # 仅对预排序的 top 150（按月动量降序）拉北向，节省大量时间
    rows.sort(key=lambda r: -(r.get("mom_mean") or -999))
    target_for_north = rows[:150]
    log(f"策略 1：拉取北向持股（仅前 150 只，节省时间）…", 75)
    north = fetch_north_holdings([r["symbol"] for r in target_for_north], 75, 95)
    for r in rows:
        n = north.get(r["symbol"], {})
        r["north_pct_now"] = n.get("north_pct_now") if n else None
        r["north_chg_20d"] = n.get("north_chg_20d") if n else None
        r["north_chg_60d"] = n.get("north_chg_60d") if n else None

    # 过滤：北向 20d 净增 > 0 且 月动量 > 0
    valid = [r for r in rows if r.get("north_chg_20d") is not None and r["north_chg_20d"] > 0 and r["mom_mean"] > 0]
    for r in valid:
        r["north_score"] = round(r["north_chg_20d"] * 1.5 + r["mom_mean"] * 0.5, 3)
    valid.sort(key=lambda x: -x["north_score"])
    other = [r for r in rows if r not in valid]
    other.sort(key=lambda x: -(x.get("mom_mean") or -999))
    final = valid + other
    df = pd.DataFrame(final)
    df.insert(0, "rank", range(1, len(df) + 1))
    log("策略 1 完成", 98)
    return df

# ─── 策略 2：量价突破 ────────────────────────────────────────────────────────

def calc_breakout(prices_map, meta, **kw):
    """突破 60 日新高 + 量能 ≥ 20日均量×1.5"""
    import pandas as pd, numpy as np
    meta_idx = meta.set_index("symbol")
    log("策略 2：量价突破 - 计算技术指标…", 60)

    rows = []
    total = len(prices_map)
    for i, (s, df) in enumerate(prices_map.items()):
        if i % 50 == 0: log(f"计算突破 {i}/{total}…", 60 + int(i / total * 30))
        if df is None or len(df) < 60: continue
        close = df["close"]
        vol = df["volume"]

        last_close = float(close.iloc[-1])
        high_60 = float(close.tail(60).max())
        # 距 60日高点（负数=未突破，0=贴顶）
        dist_high = round((last_close / high_60 - 1) * 100, 2)
        # 当日相对 60 日高点：< 1% 视为突破
        breakout = dist_high >= -1.0

        # 量比：当日 / 20 日均量
        vol_today = float(vol.iloc[-1]) if len(vol) else 0
        vol_ma20 = float(vol.tail(20).mean())
        vol_ratio = round(vol_today / vol_ma20, 2) if vol_ma20 > 0 else None

        # 5/20/60日累计
        cum5  = round(float(close.iloc[-1] / close.iloc[-6] - 1) * 100, 2)  if len(close) >= 6  else None
        cum20 = round(float(close.iloc[-1] / close.iloc[-21] - 1) * 100, 2) if len(close) >= 21 else None
        cum60 = round(float(close.iloc[-1] / close.iloc[-61] - 1) * 100, 2) if len(close) >= 61 else None

        # 30日内是否有过涨停（10% 涨幅）
        recent_limits = 0
        if len(close) >= 31:
            recent = close.tail(31)
            rets = recent.pct_change().dropna()
            recent_limits = int((rets >= 0.095).sum())

        info = meta_idx.loc[s] if s in meta_idx.index else {}
        rows.append({
            "symbol": s, "name": str(info.get("name", "")),
            "sector": str(info.get("sector", "未知")),
            "index": str(info.get("index", "")),
            "price":     round(last_close, 2),
            "high_60":   round(high_60, 2),
            "dist_high": dist_high,
            "breakout":  breakout,
            "vol_ratio": vol_ratio,
            "cum5":      cum5,
            "cum20":     cum20,
            "cum60":     cum60,
            "recent_limits": recent_limits,
        })

    # 综合：突破 + 量比≥1.5 = 强信号
    for r in rows:
        signal_strong = r["breakout"] and (r.get("vol_ratio") or 0) >= 1.5
        signal_weak   = r["breakout"]
        r["signal"]   = signal_strong
        r["watchlist"] = signal_weak
        r["score"] = (
            (3 if signal_strong else 1 if signal_weak else 0) * 30
            + max(0, r["dist_high"]) * 2
            + (r["vol_ratio"] or 0) * 5
        )

    df = pd.DataFrame(rows)
    df = df.sort_values(["signal", "watchlist", "score"], ascending=[False, False, False]).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    log("策略 2 完成", 98)
    return df

# ─── 策略 3：核心动量 ────────────────────────────────────────────────────────

def calc_momentum(prices_map, meta, **kw):
    """月度收益率动量"""
    import pandas as pd
    meta_idx = meta.set_index("symbol")
    log("策略 3：核心动量 - 计算月度统计…", 70)

    rows = []
    for s, df in prices_map.items():
        ps = _monthly_stats(df)
        if ps is None: continue
        info = meta_idx.loc[s] if s in meta_idx.index else {}
        rows.append({
            "symbol": s, "name": str(info.get("name", "")),
            "sector": str(info.get("sector", "未知")),
            "index": str(info.get("index", "")),
            **ps,
        })
    df = pd.DataFrame(rows).sort_values("mom_mean", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    log("策略 3 完成", 98)
    return df

STRATEGIES = {
    "north_money": (calc_north_money, "北向资金动量",
                    "陆股通持仓近20日净增加 + 月度动量正向，融合「聪明钱」+ 趋势"),
    "breakout":    (calc_breakout, "量价突破",
                    "突破60日新高（距≤1%）+ 量比≥1.5，A股经典趋势启动信号"),
    "momentum":    (calc_momentum, "核心动量",
                    "月度收益率均值降序，最纯粹的中线趋势策略"),
}

# ─── 筛选器 ───────────────────────────────────────────────────────────────────
# 接收 dict 参数，对策略结果做过滤：
#   pe_min/pe_max     - 市盈率区间
#   pb_min/pb_max     - 市净率区间
#   cap_min/cap_max   - 总市值区间（亿元）
#   turnover_min      - 换手率最小值 %
#   qb_min            - 量比最小值
#   exclude_st        - 排除 ST/退市股票（名称含 ST/退）
#   max_mom_dd        - 最大回撤上限（%，负数）

def apply_filters(df, filters):
    if df is None or df.empty or not filters:
        return df
    import pandas as pd
    f = filters
    cond = pd.Series(True, index=df.index)

    def _nn(col):
        return col if col in df.columns else None

    pe_col = _nn("pe")
    pb_col = _nn("pb")
    cap_col = _nn("market_cap")
    if f.get("pe_min") is not None and pe_col:
        cond &= (df[pe_col].fillna(-1) >= f["pe_min"])
    if f.get("pe_max") is not None and pe_col:
        cond &= (df[pe_col].fillna(99999) <= f["pe_max"])
    if f.get("pb_min") is not None and pb_col:
        cond &= (df[pb_col].fillna(-1) >= f["pb_min"])
    if f.get("pb_max") is not None and pb_col:
        cond &= (df[pb_col].fillna(99999) <= f["pb_max"])
    if f.get("cap_min") is not None and cap_col:
        # market_cap 单位：元，转亿元
        cond &= (df[cap_col].fillna(0) / 1e8 >= f["cap_min"])
    if f.get("cap_max") is not None and cap_col:
        cond &= (df[cap_col].fillna(0) / 1e8 <= f["cap_max"])
    if f.get("turnover_min") is not None and _nn("turnover"):
        cond &= (df["turnover"].fillna(0) >= f["turnover_min"])
    if f.get("qb_min") is not None and _nn("qb"):
        cond &= (df["qb"].fillna(0) >= f["qb_min"])
    if f.get("exclude_st") and "name" in df.columns:
        cond &= ~df["name"].fillna("").str.contains(r"ST|退|\*", regex=True)
    if f.get("max_mom_dd") is not None and _nn("max_dd"):
        # max_dd 负数；filter 期待 -20 = 最多 20% 回撤
        cond &= (df["max_dd"].fillna(-100) >= f["max_mom_dd"])
    out = df[cond].reset_index(drop=True)
    if "rank" in out.columns:
        out["rank"] = range(1, len(out) + 1)
    return out


# ─── 多维评分（A 股 4+1 维）──────────────────────────────────────────────────
# 5 个维度（0-100）：
#   logic       策略排名（top1=100, top20=5）
#   valuation   PE / PB 相对行业，越便宜越好
#   momentum    月动量 + 60日涨跌幅 + 趋势健康度
#   capital     量比 + 换手率 + 北向持股变化（资金面）
#   tech        距 60日高 + Sharpe + 最大回撤

def _clamp(v, lo=0, hi=100):
    return max(lo, min(hi, v))

def _score_row(rank, total, row):
    import pandas as pd
    # 1) logic
    logic = _clamp(100 - (rank - 1) * (95 / max(total - 1, 1))) if rank else 50

    # 2) valuation: PE / PB 越低越好（但负 PE 视为不计入）
    v_parts = []
    pe = row.get("pe")
    pb = row.get("pb")
    if pe is not None and pe > 0:
        # PE 5→100, 20→60, 40→30, 80+→0
        v_parts.append(_clamp(110 - pe * 1.4))
    if pb is not None and pb > 0:
        # PB 1→100, 3→60, 5→30, 10+→0
        v_parts.append(_clamp(110 - pb * 12))
    valuation = round(sum(v_parts) / len(v_parts), 1) if v_parts else 50.0

    # 3) momentum: 月动量 + 60日涨跌
    m_parts = []
    mom = row.get("mom_mean")
    if mom is not None:
        # -5%→0, 0→50, 5%→90
        m_parts.append(_clamp(50 + mom * 8))
    r60 = row.get("ret_60d")
    if r60 is not None:
        # -20%→0, 0→50, +30%→90
        m_parts.append(_clamp(50 + r60 * 1.5))
    tr = row.get("total_ret")
    if tr is not None:
        m_parts.append(_clamp(50 + tr * 0.8))
    momentum_s = round(sum(m_parts) / len(m_parts), 1) if m_parts else 50.0

    # 4) capital：量比 + 换手率 + 北向持股变化
    c_parts = []
    qb = row.get("qb")
    if qb is not None and qb > 0:
        # 0.8→30, 1.0→50, 1.5→80, 2+→95
        c_parts.append(_clamp(20 + qb * 35))
    to = row.get("turnover")
    if to is not None:
        # 1%→30, 3%→60, 5%→80, 10%+→95
        c_parts.append(_clamp(20 + to * 8))
    nc = row.get("north_chg_20d")
    if nc is not None:
        # 0→50, +0.3%→80, -0.3%→20
        c_parts.append(_clamp(50 + nc * 100))
    capital_s = round(sum(c_parts) / len(c_parts), 1) if c_parts else 50.0

    # 5) tech：距60日高 + Sharpe + 最大回撤
    t_parts = []
    dh = row.get("dist_high")  # breakout 策略才有，距 60日高 %
    if dh is not None:
        # 0→100, -5%→60, -15%→20
        t_parts.append(_clamp(100 + dh * 4))
    sh = row.get("sharpe")
    if sh is not None:
        # 0→40, 1→75, 2+→95
        t_parts.append(_clamp(40 + sh * 35))
    md = row.get("max_dd")
    if md is not None:
        # -5%→90, -15%→60, -30%→20
        t_parts.append(_clamp(100 + md * 3))
    vr = row.get("vol_ratio")
    if vr is not None:
        t_parts.append(_clamp(30 + vr * 25))
    tech_s = round(sum(t_parts) / len(t_parts), 1) if t_parts else 50.0

    # 综合：权重
    composite = round(
        0.30 * logic +
        0.20 * valuation +
        0.20 * momentum_s +
        0.15 * capital_s +
        0.15 * tech_s, 1)

    return {
        "score_total":     composite,
        "score_logic":     round(logic, 1),
        "score_valuation": valuation,
        "score_momentum":  momentum_s,
        "score_capital":   capital_s,
        "score_tech":      tech_s,
    }

def add_scores(df, top_n=20):
    if df is None or df.empty:
        return df
    for col in ("score_total","score_logic","score_valuation",
                "score_momentum","score_capital","score_tech"):
        df[col] = None
    total = min(top_n, len(df))
    for i in range(total):
        idx = df.index[i]
        rank = int(df.at[idx, "rank"]) if "rank" in df.columns else i + 1
        row = df.iloc[i].to_dict()
        scores = _score_row(rank, total, row)
        for k, v in scores.items():
            df.at[idx, k] = v
    return df


# ─── 市场温度（A 股版）──────────────────────────────────────────────────────

_market_cache = {"ts": 0, "data": None}
_MARKET_TTL = 3600

def fetch_market_temp():
    """A 股市场温度指标
    1. 北向资金 5/20日净流入
    2. 沪深两市成交额
    3. 涨停 / 跌停家数
    4. 沪深300 PE 分位
    5. 融资融券余额
    6. 上证 / 创业板 / 科创50 指数动量
    """
    import akshare as ak
    import pandas as pd, numpy as np

    indicators = {}

    # 1. 沪深300 近5日动量（替代北向资金 — 港交所自 2024-08 起停止披露北向日数据）
    try:
        log("市场温度：沪深300 动量…", 10)
        df = ak.stock_zh_index_daily_em(symbol="sh000300")
        df = df.set_index(pd.to_datetime(df["date"])).sort_index()
        close60 = df["close"].tail(60)
        last = float(close60.iloc[-1])
        chg5  = round((last/float(close60.iloc[-6 ]) - 1)*100, 2) if len(close60) >= 6  else None
        chg20 = round((last/float(close60.iloc[-21]) - 1)*100, 2) if len(close60) >= 21 else None
        # 历史 5 日滚动收益（用于分位）
        rolling5 = (close60 / close60.shift(5) - 1) * 100
        rolling5 = rolling5.dropna()
        hist = [round(float(v), 2) for v in rolling5.tolist()]
        dates = [str(d.date()) for d in rolling5.index]
        indicators["hs300_mom"] = {
            "value": chg5,
            "value_label": f"沪深300 近5日 {chg5:+}%",
            "extra": f"近20日 {chg20:+}% · 价格 {last:.0f}",
            "history": hist, "dates": dates,
            "percentile": _pct(hist, chg5),
            "warn": "low",
        }
    except Exception as e:
        log(f"沪深300 动量失败：{e}", -1)

    # 2. 两市成交额
    try:
        log("市场温度：两市成交额…", 25)
        sh = ak.stock_zh_index_daily_em(symbol="sh000001")
        sz = ak.stock_zh_index_daily_em(symbol="sz399001")
        sh = sh.set_index(pd.to_datetime(sh["date"])).sort_index()
        sz = sz.set_index(pd.to_datetime(sz["date"])).sort_index()
        # amount 单位是元
        total_amount = (sh["amount"] + sz["amount"]).dropna().tail(60) / 1e8  # 亿元
        val = round(float(total_amount.iloc[-1]), 0)
        chg = round(float(total_amount.iloc[-1] - total_amount.iloc[-2]), 0) if len(total_amount) >= 2 else None
        hist = [round(float(v), 0) for v in total_amount.tolist()]
        dates = [str(d.date()) for d in total_amount.index]
        indicators["amount"] = {
            "value": val, "change": chg,
            "value_label": f"两市成交 {val:,.0f} 亿",
            "extra": "情绪指标：高成交=活跃，低成交=观望",
            "history": hist, "dates": dates,
            "percentile": _pct(hist, val),
            "warn": "neutral",
        }
    except Exception as e:
        log(f"成交额失败：{e}", -1)

    # 3. 涨停 / 跌停家数
    try:
        log("市场温度：涨跌停统计…", 40)
        zt = ak.stock_zt_pool_em(date=datetime.now().strftime("%Y%m%d"))
        dt = ak.stock_zt_pool_dtgc_em(date=datetime.now().strftime("%Y%m%d"))
        zt_cnt = len(zt) if zt is not None else 0
        dt_cnt = len(dt) if dt is not None else 0
        net = zt_cnt - dt_cnt
        indicators["zt_dt"] = {
            "value": net,
            "value_label": f"涨停 {zt_cnt} / 跌停 {dt_cnt}",
            "extra": f"净额 {net} 家（正=多头强势）",
            "percentile": None, "history": [], "dates": [],
            "warn": "low",
        }
    except Exception as e:
        log(f"涨跌停失败：{e}", -1)

    # 4. 沪深 300 PE 历史分位（10 年）
    try:
        log("市场温度：沪深300 估值…", 55)
        df = ak.stock_index_pe_lg(symbol="沪深300")
        # 列名包含: 日期 / 指数 / 静态市盈率 / 静态市盈率分位数 / 滚动市盈率 / 滚动市盈率分位数
        col_date = next(c for c in df.columns if "日期" in c)
        col_pe   = next((c for c in df.columns if c == "静态市盈率"), None) \
                  or next((c for c in df.columns if "静态市盈率" in c and "分位" not in c and "等权" not in c), None) \
                  or next(c for c in df.columns if "市盈率" in c and "分位" not in c)
        col_pct = next((c for c in df.columns if "分位" in c and "静态" in c), None) \
                 or next((c for c in df.columns if "分位" in c), None)
        df = df.rename(columns={col_date: "date", col_pe: "pe"})
        if col_pct:
            df = df.rename(columns={col_pct: "pe_pct"})
        df["date"] = pd.to_datetime(df["date"])
        df["pe"] = pd.to_numeric(df["pe"], errors="coerce")
        df = df.dropna(subset=["pe"]).set_index("date").sort_index()
        pe_now = float(df["pe"].iloc[-1])
        # 优先用 akshare 自己的分位数，否则自算 10 年
        if "pe_pct" in df.columns:
            df["pe_pct"] = pd.to_numeric(df["pe_pct"], errors="coerce")
            akpct = float(df["pe_pct"].iloc[-1]) if not pd.isna(df["pe_pct"].iloc[-1]) else None
            pct = round(akpct, 1) if akpct is not None else _pct([float(v) for v in df["pe"].tail(2520).tolist()], pe_now)
        else:
            pct = _pct([float(v) for v in df["pe"].tail(2520).tolist()], pe_now)
        hist_60d = [round(float(v), 2) for v in df["pe"].tail(60).tolist()]
        dates60 = [str(d.date()) for d in df.tail(60).index]
        indicators["hs300_pe"] = {
            "value": round(pe_now, 1),
            "value_label": f"沪深300 PE {pe_now:.1f}",
            "extra": f"10年分位 {pct}%",
            "history": hist_60d, "dates": dates60,
            "percentile": pct,
            "warn": "high",
        }
    except Exception as e:
        log(f"PE 失败：{e}", -1)

    # 5. 融资融券余额
    try:
        log("市场温度：融资融券…", 70)
        df_total = ak.stock_margin_account_info()
        col_date = next(c for c in df_total.columns if "日期" in c)
        col_rz   = next(c for c in df_total.columns if "融资余额" in c)
        col_rq   = next(c for c in df_total.columns if "融券余额" in c)
        df_total = df_total.rename(columns={col_date: "date", col_rz: "rz", col_rq: "rq"})
        df_total["date"] = pd.to_datetime(df_total["date"])
        df_total["total"] = pd.to_numeric(df_total["rz"], errors="coerce") + pd.to_numeric(df_total["rq"], errors="coerce")
        df_total = df_total.dropna(subset=["total"]).set_index("date").sort_index().tail(60)
        # 该接口余额单位是「亿元」
        val = round(float(df_total["total"].iloc[-1]), 0)
        chg20 = round(float(df_total["total"].iloc[-1] - df_total["total"].iloc[-21]), 0) if len(df_total) >= 21 else None
        hist = [round(float(v), 0) for v in df_total["total"].tolist()]
        dates = [str(d.date()) for d in df_total.index]
        indicators["margin"] = {
            "value": val,
            "value_label": f"两融余额 {val} 亿",
            "extra": f"近20日变化 {chg20:+} 亿" if chg20 is not None else "",
            "history": hist, "dates": dates,
            "percentile": _pct(hist, val),
            "warn": "high",
        }
    except Exception as e:
        log(f"两融失败：{e}", -1)

    # 6. 主要指数动量（上证 / 创业板 / 科创）
    try:
        log("市场温度：指数动量…", 85)
        idx_data = []
        for sym, name in [("sh000001","上证"), ("sz399006","创业板"), ("sh000688","科创50")]:
            try:
                d = ak.stock_zh_index_daily_em(symbol=sym)
                d = d.set_index(pd.to_datetime(d["date"])).sort_index().tail(20)
                ret = float(d["close"].iloc[-1] / d["close"].iloc[0] - 1) * 100
                idx_data.append(f"{name} {ret:+.2f}%")
            except: continue
        if idx_data:
            indicators["indices"] = {
                "value": None,
                "value_label": "近20日指数",
                "extra": "  ·  ".join(idx_data),
                "history": [], "dates": [], "percentile": None,
                "warn": "neutral",
            }
    except Exception as e:
        log(f"指数失败：{e}", -1)

    return indicators

def _pct(arr, v):
    if not arr or v is None: return None
    arr = [a for a in arr if a is not None]
    if not arr: return None
    import numpy as np
    return round(float(np.sum(np.array(arr) <= v) / len(arr) * 100), 1)

@app.route("/api/market")
def api_market():
    now = time.time()
    if _market_cache["data"] and now - _market_cache["ts"] < _MARKET_TTL:
        return jsonify(_market_cache["data"])
    try:
        data = fetch_market_temp()
        _market_cache.update(ts=now, data=data)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ─── 主计算 ───────────────────────────────────────────────────────────────────

_calc_lock = threading.Lock()

@app.route("/api/run", methods=["POST"])
def api_run():
    body = request.json or {}

    def worker():
        try:
            import pandas as pd
            scope = body.get("universe", "hs300_zz500")
            strategy = body.get("strategy", "north_money")
            months = int(body.get("months", 6))

            meta = fetch_universe(scope)
            sector_filter = body.get("sector_filter", "")
            if sector_filter and sector_filter != "all":
                meta = meta[meta["sector"].str.contains(sector_filter, na=False)]
            symbols = meta["symbol"].astype(str).tolist()
            log(f"股票池 {len(symbols)} 只 · 策略 {strategy} · 区间 {months} 个月", 18)

            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=months * 31)).strftime("%Y-%m-%d")
            prices_map = load_prices(symbols, start_date, end_date)
            log(f"成功下载 {len(prices_map)}/{len(symbols)} 只行情", 55)

            # 行情下载失败率高（>50%）→ 明确报错而非让策略静默跑空
            if len(prices_map) == 0:
                _broadcast("error", {"msg": f"无法连接 akshare 数据源（0/{len(symbols)} 只成功）— 可能是网络不稳或东方财富接口限频，稍后重试"})
                return
            if len(prices_map) < len(symbols) * 0.3:
                log(f"⚠ 仅 {len(prices_map)}/{len(symbols)} 只下载成功，结果可能不完整", 56)

            fn = STRATEGIES.get(strategy, STRATEGIES["north_money"])[0]
            df = fn(prices_map, meta)

            # 把 meta 里的估值/市值/量比合并到结果（覆盖式 left join）
            try:
                merge_cols = [c for c in ("symbol","pe","pb","market_cap","circ_cap","turnover","qb","ret_60d") if c in meta.columns]
                if "symbol" in df.columns and merge_cols:
                    df = df.merge(meta[merge_cols], on="symbol", how="left", suffixes=("", "_meta"))
            except Exception as e:
                log(f"meta 合并失败：{e}", -1)

            # 应用筛选器（在评分之前，免得评分稀疏）
            filters = body.get("filters") or {}
            before = len(df)
            df = apply_filters(df, filters)
            after = len(df)
            if filters:
                log(f"筛选：{before} → {after} 只", 96)

            # 多维评分（前 20 名）
            df = add_scores(df, top_n=20)

            rows = df.head(100).to_dict(orient="records")
            _broadcast("done", {"rows": rows, "strategy": strategy,
                                "start": start_date, "end": end_date,
                                "filtered_in": after, "filtered_total": before})
        except Exception as e:
            import traceback
            traceback.print_exc()
            _broadcast("error", {"msg": str(e)})

    if not _calc_lock.acquire(blocking=False):
        return jsonify({"error": "已有任务运行中，请稍候"}), 429

    def run_release():
        try: worker()
        finally: _calc_lock.release()

    threading.Thread(target=run_release, daemon=True).start()
    return jsonify({"ok": True})

@app.route("/api/sectors")
def api_sectors():
    scope = request.args.get("universe", "hs300_zz500")
    try:
        meta = fetch_universe(scope)
        sectors = sorted(meta["sector"].dropna().unique().tolist())
        return jsonify({"sectors": sectors})
    except Exception as e:
        return jsonify({"sectors": [], "error": str(e)})

@app.route("/api/strategies")
def api_strategies():
    return jsonify({k: {"name": v[1], "desc": v[2]} for k, v in STRATEGIES.items()})

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
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "astock_ui.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5002))
    print(f"\n  A 股多策略排序工具")
    print(f"  访问：http://localhost:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
