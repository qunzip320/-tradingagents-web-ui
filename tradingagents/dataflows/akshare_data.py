"""AkShare 数据供应商 —— 专为 A股设计。

支持上交所（.SS）和深交所（.SZ）格式的 ticker，
自动剥离后缀后调用 akshare 接口。

所有函数签名与 y_finance.py 保持一致，可直接注册到 interface.py。
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Annotated

logger = logging.getLogger(__name__)


def _clean_symbol(ticker: str) -> str:
    """600519.SS → 600519，000858.SZ → 000858"""
    return ticker.split(".")[0]


def _is_a_share(ticker: str) -> bool:
    suffix = ticker.upper().split(".")[-1] if "." in ticker else ""
    return suffix in ("SS", "SZ", "BJ") or (ticker.isdigit() and len(ticker) == 6)


# ─────────────────────────────────────────────
# 1. OHLCV 历史行情
# ─────────────────────────────────────────────

def get_stock(
    symbol: Annotated[str, "股票代码，如 600519.SS"],
    start_date: Annotated[str, "开始日期 YYYY-MM-DD"],
    end_date: Annotated[str, "结束日期 YYYY-MM-DD"],
) -> str:
    """获取 A股 OHLCV 历史行情（前复权）。"""
    try:
        import akshare as ak
        code = _clean_symbol(symbol)
        sd = start_date.replace("-", "")
        ed = end_date.replace("-", "")
        # 优先用新浪接口，更稳定
        try:
            df = ak.stock_zh_a_daily(symbol=f"sh{code}" if code.startswith("6") else f"sz{code}",
                                      start_date=sd, end_date=ed, adjust="qfq")
            if df is not None and not df.empty:
                df = df.reset_index()
                col_map = {"date": "Date", "open": "Open", "close": "Close",
                           "high": "High", "low": "Low", "volume": "Volume"}
                df = df.rename(columns=col_map)
                return f"# {symbol} 历史行情（前复权）{start_date} ~ {end_date}\n" + df.to_csv(index=False)
        except Exception:
            pass
        # 回退到东方财富接口
        df = ak.stock_zh_a_hist(symbol=code, period="daily",
                                 start_date=sd, end_date=ed, adjust="qfq")
        if df is None or df.empty:
            return f"未找到 {symbol} 在 {start_date} 至 {end_date} 的行情数据"
        col_map = {
            "日期": "Date", "开盘": "Open", "收盘": "Close",
            "最高": "High", "最低": "Low", "成交量": "Volume",
            "成交额": "Amount", "涨跌幅": "PctChange", "换手率": "Turnover",
        }
        df = df.rename(columns=col_map)
        return f"# {symbol} 历史行情（前复权）{start_date} ~ {end_date}\n" + df.to_csv(index=False)
    except Exception as e:
        logger.warning("akshare get_stock failed for %s: %s", symbol, e)
        return f"<akshare 行情数据获取失败: {e}>"


# ─────────────────────────────────────────────
# 2. 技术指标（复用 stockstats，数据来自 akshare）
# ─────────────────────────────────────────────

def get_indicators(
    symbol: Annotated[str, "股票代码"],
    indicator: Annotated[str, "指标名称，如 rsi, macd, close_50_sma"],
    curr_date: Annotated[str, "当前日期 YYYY-MM-DD"],
    look_back_days: Annotated[int, "回溯天数"] = 30,
) -> str:
    """基于 akshare 行情数据计算技术指标。"""
    try:
        import akshare as ak
        import pandas as pd
        from stockstats import StockDataFrame

        code = _clean_symbol(symbol)
        end_dt = datetime.strptime(curr_date, "%Y-%m-%d")
        start_dt = end_dt - timedelta(days=look_back_days + 60)  # 多取一些用于计算
        sd = start_dt.strftime("%Y%m%d")
        ed = end_dt.strftime("%Y%m%d")

        df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=sd, end_date=ed, adjust="qfq")
        if df is None or df.empty:
            return f"<无法获取 {symbol} 的行情数据>"

        df = df.rename(columns={"日期": "date", "开盘": "open", "收盘": "close",
                                  "最高": "high", "最低": "low", "成交量": "volume"})
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()

        stock = StockDataFrame.retype(df.copy())
        if indicator not in stock.columns:
            _ = stock[indicator]  # 触发计算

        result = stock[[indicator]].tail(look_back_days)
        result = result[result.index <= pd.Timestamp(curr_date)]
        return f"# {symbol} {indicator} 指标（最近 {look_back_days} 天）\n" + result.to_csv()
    except Exception as e:
        logger.warning("akshare get_indicators failed for %s/%s: %s", symbol, indicator, e)
        return f"<akshare 技术指标获取失败: {e}>"


# ─────────────────────────────────────────────
# 3. 基本面数据
# ─────────────────────────────────────────────

def get_fundamentals(
    ticker: Annotated[str, "股票代码"],
    curr_date: Annotated[str, "当前日期 YYYY-MM-DD"] = None,
) -> str:
    """获取 A股公司基本面信息。"""
    try:
        import akshare as ak
        code = _clean_symbol(ticker)
        lines = [f"# {ticker} 基本面信息"]

        # 接口1：东方财富个股信息
        try:
            info = ak.stock_individual_info_em(symbol=code)
            for _, row in info.iterrows():
                lines.append(f"{row.iloc[0]}: {row.iloc[1]}")
        except Exception as e1:
            logger.debug("stock_individual_info_em failed: %s", e1)
            # 回退：用 yfinance 获取基本信息
            try:
                import yfinance as yf
                t = yf.Ticker(ticker)
                info = t.info
                for k in ["longName", "sector", "industry", "marketCap", "trailingPE",
                          "forwardPE", "dividendYield", "returnOnEquity", "debtToEquity"]:
                    if info.get(k) is not None:
                        lines.append(f"{k}: {info[k]}")
            except Exception:
                pass

        # 财务指标（最近4期）
        try:
            year = (datetime.strptime(curr_date, "%Y-%m-%d").year if curr_date else datetime.now().year)
            fin = ak.stock_financial_analysis_indicator(symbol=code, start_year=str(year - 1))
            if fin is not None and not fin.empty:
                lines.append("\n## 财务指标（近期）")
                key_cols = ["日期", "净资产收益率(%)", "总资产净利润率(%)", "净利润率(%)",
                            "资产负债率(%)", "流动比率", "速动比率", "每股收益(元)"]
                available = [c for c in key_cols if c in fin.columns]
                lines.append(fin[available].head(4).to_string(index=False))
        except Exception as fe:
            logger.debug("akshare financial indicator failed: %s", fe)

        return "\n".join(lines)
    except Exception as e:
        logger.warning("akshare get_fundamentals failed for %s: %s", ticker, e)
        return f"<akshare 基本面数据获取失败: {e}>"


def get_balance_sheet(
    ticker: Annotated[str, "股票代码"],
) -> str:
    """获取 A股资产负债表。"""
    try:
        import akshare as ak
        code = _clean_symbol(ticker)
        # 使用利润表接口获取资产负债相关数据
        df = ak.stock_financial_analysis_indicator(symbol=code, start_year=str(datetime.now().year - 2))
        if df is None or df.empty:
            return f"<未找到 {ticker} 的资产负债数据>"
        bs_cols = [c for c in df.columns if any(k in c for k in ["资产", "负债", "权益", "货币"])]
        available = ["日期"] + bs_cols[:15] if "日期" in df.columns else bs_cols[:15]
        return f"# {ticker} 资产负债相关指标\n" + df[available].head(8).to_csv(index=False)
    except Exception as e:
        logger.warning("akshare get_balance_sheet failed for %s: %s", ticker, e)
        return f"<akshare 资产负债表获取失败: {e}>"


def get_cashflow(
    ticker: Annotated[str, "股票代码"],
) -> str:
    """获取 A股现金流数据。"""
    try:
        import akshare as ak
        code = _clean_symbol(ticker)
        df = ak.stock_financial_analysis_indicator(symbol=code, start_year=str(datetime.now().year - 2))
        if df is None or df.empty:
            return f"<未找到 {ticker} 的现金流数据>"
        cf_cols = [c for c in df.columns if any(k in c for k in ["现金", "经营", "投资", "筹资"])]
        available = ["日期"] + cf_cols[:15] if "日期" in df.columns else cf_cols[:15]
        return f"# {ticker} 现金流相关指标\n" + df[available].head(8).to_csv(index=False)
    except Exception as e:
        logger.warning("akshare get_cashflow failed for %s: %s", ticker, e)
        return f"<akshare 现金流数据获取失败: {e}>"


def get_income_statement(
    ticker: Annotated[str, "股票代码"],
) -> str:
    """获取 A股利润表数据。"""
    try:
        import akshare as ak
        code = _clean_symbol(ticker)
        df = ak.stock_financial_analysis_indicator(symbol=code, start_year=str(datetime.now().year - 2))
        if df is None or df.empty:
            return f"<未找到 {ticker} 的利润表数据>"
        inc_cols = [c for c in df.columns if any(k in c for k in ["利润", "收入", "营业", "毛利", "净利"])]
        available = ["日期"] + inc_cols[:15] if "日期" in df.columns else inc_cols[:15]
        return f"# {ticker} 利润表相关指标\n" + df[available].head(8).to_csv(index=False)
    except Exception as e:
        logger.warning("akshare get_income_statement failed for %s: %s", ticker, e)
        return f"<akshare 利润表数据获取失败: {e}>"


def get_insider_transactions(
    ticker: Annotated[str, "股票代码"],
) -> str:
    """获取 A股大股东增减持记录。"""
    try:
        import akshare as ak
        code = _clean_symbol(ticker)
        # 大股东增减持
        df = ak.stock_em_hold_stock_detail(symbol=code)
        if df is None or df.empty:
            return f"<未找到 {ticker} 的大股东增减持记录>"
        return f"# {ticker} 大股东增减持记录\n" + df.head(20).to_csv(index=False)
    except Exception as e:
        logger.warning("akshare get_insider_transactions failed for %s: %s", ticker, e)
        return f"<akshare 增减持数据获取失败: {e}>"


# ─────────────────────────────────────────────
# 4. 新闻数据
# ─────────────────────────────────────────────

def get_news(
    ticker: Annotated[str, "股票代码"],
    start_date: Annotated[str, "开始日期 YYYY-MM-DD"],
    end_date: Annotated[str, "结束日期 YYYY-MM-DD"],
) -> str:
    """获取 A股个股新闻。"""
    try:
        import akshare as ak
        code = _clean_symbol(ticker)
        results = []

        # 接口1：东方财富个股新闻
        try:
            df = ak.stock_news_em(symbol=code)
            if df is not None and not df.empty:
                date_col = next((c for c in ["发布时间", "时间", "date"] if c in df.columns), None)
                if date_col:
                    df[date_col] = df[date_col].astype(str).str[:10]
                    df = df[(df[date_col] >= start_date) & (df[date_col] <= end_date)]
                for _, row in df.head(15).iterrows():
                    title = row.get("新闻标题", row.get("title", ""))
                    time_ = row.get("发布时间", row.get("时间", ""))
                    if title:
                        results.append(f"[{str(time_)[:10]}] {title}")
        except Exception:
            pass

        # 接口2：同花顺个股资讯
        if not results:
            try:
                df2 = ak.stock_info_global_ths()
                if df2 is not None and not df2.empty:
                    for _, row in df2.head(10).iterrows():
                        title = str(row.iloc[0]) if len(row) > 0 else ""
                        if title and code in title or ticker.split(".")[0] in title:
                            results.append(f"[最新] {title}")
            except Exception:
                pass

        if not results:
            return f"<{ticker} 在 {start_date} 至 {end_date} 期间无新闻数据>"

        lines = [f"# {ticker} 个股新闻 {start_date} ~ {end_date}"] + results
        return "\n".join(lines)
    except Exception as e:
        logger.warning("akshare get_news failed for %s: %s", ticker, e)
        return f"<akshare 新闻数据获取失败: {e}>"


def get_global_news(
    start_date: Annotated[str, "开始日期 YYYY-MM-DD"],
    end_date: Annotated[str, "结束日期 YYYY-MM-DD"],
    queries: Annotated[list, "搜索关键词列表"] = None,
    limit: Annotated[int, "最大文章数"] = 10,
) -> str:
    """获取 A股市场宏观新闻（财经头条）。"""
    try:
        import akshare as ak
        df = ak.stock_news_main_sina()
        if df is None or df.empty:
            return "<无法获取宏观财经新闻>"
        lines = [f"# 宏观财经新闻"]
        for _, row in df.head(limit).iterrows():
            title = row.get("标题", row.get("title", ""))
            time_ = row.get("时间", row.get("time", ""))
            lines.append(f"[{time_}] {title}")
        return "\n".join(lines)
    except Exception as e:
        logger.warning("akshare get_global_news failed: %s", e)
        return f"<akshare 宏观新闻获取失败: {e}>"
