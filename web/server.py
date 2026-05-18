import asyncio
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator

from dotenv import load_dotenv
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, JSONResponse
from sse_starlette.sse import EventSourceResponse

load_dotenv(Path(__file__).parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).parent.parent))

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.graph.analyst_execution import build_analyst_execution_plan

app = FastAPI(title="TradingAgents Web")

STATIC_DIR = Path(__file__).parent / "static"
REPORTS_DIR = Path(__file__).parent.parent / "reports"

ANALYST_ORDER = ["market", "social", "news", "fundamentals"]
ANALYST_AGENT_NAMES = {
    "market": "Market Analyst",
    "social": "Sentiment Analyst",
    "news": "News Analyst",
    "fundamentals": "Fundamentals Analyst",
}
ANALYST_REPORT_MAP = {
    "market": "market_report",
    "social": "sentiment_report",
    "news": "news_report",
    "fundamentals": "fundamentals_report",
}


def _build_config() -> dict:
    config = DEFAULT_CONFIG.copy()
    config["output_language"] = "Chinese"
    config["max_debate_rounds"] = 1
    config["max_risk_discuss_rounds"] = 1
    return config


def _classify_message(message) -> tuple[str, str | None]:
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    content = getattr(message, "content", None)
    if isinstance(content, list):
        parts = [
            item.get("text", "").strip()
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        content = " ".join(p for p in parts if p) or None
    elif isinstance(content, str):
        content = content.strip() or None
    else:
        content = None

    if isinstance(message, HumanMessage):
        return ("Control" if content == "Continue" else "User", content)
    if isinstance(message, ToolMessage):
        return ("Data", content)
    if isinstance(message, AIMessage):
        return ("Agent", content)
    return ("System", content)


async def _stream_analysis(ticker: str, date: str, analysts: list[str]) -> AsyncGenerator[str, None]:
    def emit(data: dict) -> str:
        return json.dumps(data, ensure_ascii=False)

    config = _build_config()
    start_time = time.time()

    try:
        graph = TradingAgentsGraph(
            selected_analysts=analysts,
            debug=False,
            config=config,
        )
    except Exception as e:
        yield emit({"type": "error", "message": f"初始化失败: {e}"})
        return

    # 初始化所有 agent 状态
    all_agents = [ANALYST_AGENT_NAMES[a] for a in analysts if a in ANALYST_AGENT_NAMES]
    all_agents += ["Bull Researcher", "Bear Researcher", "Research Manager",
                   "Trader", "Aggressive Analyst", "Conservative Analyst",
                   "Neutral Analyst", "Portfolio Manager"]
    for agent in all_agents:
        yield emit({"type": "agent_status", "agent": agent, "status": "pending"})

    # 第一个分析师设为 in_progress
    if analysts:
        first = ANALYST_AGENT_NAMES.get(analysts[0])
        if first:
            yield emit({"type": "agent_status", "agent": first, "status": "in_progress"})

    await asyncio.sleep(0)

    try:
        init_state = graph.propagator.create_initial_state(ticker, date, asset_type="stock")
        args = graph.propagator.get_graph_args()
    except Exception as e:
        yield emit({"type": "error", "message": f"状态初始化失败: {e}"})
        return

    report_sections: dict[str, str | None] = {
        ANALYST_REPORT_MAP[a]: None for a in analysts if a in ANALYST_REPORT_MAP
    }
    report_sections.update({
        "investment_plan": None,
        "trader_investment_plan": None,
        "final_trade_decision": None,
    })

    processed_ids: set = set()
    found_active = False
    research_started = False

    def run_stream():
        return list(graph.graph.stream(init_state, **args))

    loop = asyncio.get_event_loop()

    try:
        chunks = await loop.run_in_executor(None, run_stream)
    except Exception as e:
        yield emit({"type": "error", "message": f"分析执行失败: {str(e)}"})
        return

    # 逐 chunk 处理并推送
    for chunk in chunks:
        # 消息
        for message in chunk.get("messages", []):
            msg_id = getattr(message, "id", None)
            if msg_id:
                if msg_id in processed_ids:
                    continue
                processed_ids.add(msg_id)
            msg_type, content = _classify_message(message)
            if content and content.strip() and msg_type in ("Agent", "Data", "Tool"):
                short = content[:300] + ("..." if len(content) > 300 else "")
                yield emit({"type": "message", "msg_type": msg_type, "content": short})

            if hasattr(message, "tool_calls") and message.tool_calls:
                for tc in message.tool_calls:
                    name = tc["name"] if isinstance(tc, dict) else tc.name
                    yield emit({"type": "message", "msg_type": "Tool", "content": f"调用工具: {name}"})

        # 分析师报告
        found_active = False
        for analyst_key in ANALYST_ORDER:
            if analyst_key not in analysts:
                continue
            agent_name = ANALYST_AGENT_NAMES[analyst_key]
            report_key = ANALYST_REPORT_MAP[analyst_key]
            if chunk.get(report_key):
                report_sections[report_key] = chunk[report_key]
                yield emit({"type": "report", "section": report_key, "content": chunk[report_key]})
                yield emit({"type": "agent_status", "agent": agent_name, "status": "completed"})
            elif report_sections.get(report_key):
                yield emit({"type": "agent_status", "agent": agent_name, "status": "completed"})
            elif not found_active:
                yield emit({"type": "agent_status", "agent": agent_name, "status": "in_progress"})
                found_active = True

        # 研究团队
        if chunk.get("investment_debate_state"):
            debate = chunk["investment_debate_state"]
            if not research_started and (debate.get("bull_history") or debate.get("bear_history")):
                research_started = True
                yield emit({"type": "agent_status", "agent": "Bull Researcher", "status": "in_progress"})
                yield emit({"type": "agent_status", "agent": "Bear Researcher", "status": "in_progress"})

            if debate.get("bull_history"):
                yield emit({"type": "debate", "section": "bull", "content": debate["bull_history"]})
            if debate.get("bear_history"):
                yield emit({"type": "debate", "section": "bear", "content": debate["bear_history"]})
            if debate.get("judge_decision"):
                report_sections["investment_plan"] = debate["judge_decision"]
                yield emit({"type": "report", "section": "investment_plan", "content": debate["judge_decision"]})
                yield emit({"type": "agent_status", "agent": "Bull Researcher", "status": "completed"})
                yield emit({"type": "agent_status", "agent": "Bear Researcher", "status": "completed"})
                yield emit({"type": "agent_status", "agent": "Research Manager", "status": "completed"})
                yield emit({"type": "agent_status", "agent": "Trader", "status": "in_progress"})

        # 交易员
        if chunk.get("trader_investment_plan"):
            report_sections["trader_investment_plan"] = chunk["trader_investment_plan"]
            yield emit({"type": "report", "section": "trader_investment_plan", "content": chunk["trader_investment_plan"]})
            yield emit({"type": "agent_status", "agent": "Trader", "status": "completed"})
            yield emit({"type": "agent_status", "agent": "Aggressive Analyst", "status": "in_progress"})

        # 风险管理
        if chunk.get("risk_debate_state"):
            risk = chunk["risk_debate_state"]
            if risk.get("aggressive_history"):
                yield emit({"type": "agent_status", "agent": "Aggressive Analyst", "status": "in_progress"})
            if risk.get("conservative_history"):
                yield emit({"type": "agent_status", "agent": "Conservative Analyst", "status": "in_progress"})
            if risk.get("neutral_history"):
                yield emit({"type": "agent_status", "agent": "Neutral Analyst", "status": "in_progress"})
            if risk.get("judge_decision"):
                report_sections["final_trade_decision"] = risk["judge_decision"]
                yield emit({"type": "report", "section": "final_trade_decision", "content": risk["judge_decision"]})
                for a in ["Aggressive Analyst", "Conservative Analyst", "Neutral Analyst", "Portfolio Manager"]:
                    yield emit({"type": "agent_status", "agent": a, "status": "completed"})

        # 最终决策
        if chunk.get("final_trade_decision"):
            decision_text = chunk["final_trade_decision"]
            decision = graph.process_signal(decision_text)
            yield emit({"type": "final", "decision": decision, "content": decision_text})
            for agent in all_agents:
                yield emit({"type": "agent_status", "agent": agent, "status": "completed"})

        await asyncio.sleep(0)

    elapsed = time.time() - start_time
    yield emit({"type": "done", "elapsed": round(elapsed, 1)})


@app.get("/")
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/analyze")
async def analyze(
    ticker: str = Query(..., description="股票代码"),
    date: str = Query(default="", description="分析日期 YYYY-MM-DD"),
    analysts: str = Query(default="market,social,news,fundamentals"),
):
    if not date:
        date = datetime.now().strftime("%Y-%m-%d")
    analyst_list = [a.strip() for a in analysts.split(",") if a.strip() in ANALYST_ORDER]
    if not analyst_list:
        analyst_list = list(ANALYST_ORDER)

    async def generator():
        async for event in _stream_analysis(ticker.upper(), date, analyst_list):
            yield {"data": event}

    return EventSourceResponse(generator())


# ═══════════════════════════════════════════════════════
# 历史报告 API
# ═══════════════════════════════════════════════════════

SECTION_NAMES: dict[str, str] = {
    "market_report": "市场分析",
    "sentiment_report": "情感分析",
    "news_report": "新闻分析",
    "fundamentals_report": "基本面分析",
    "investment_plan": "研究决策",
    "trader_investment_plan": "交易计划",
    "final_trade_decision": "最终决策",
}

SECTION_FILES: dict[str, str] = {
    "market": "1_analysts/market.md",
    "sentiment": "1_analysts/sentiment.md",
    "news": "1_analysts/news.md",
    "fundamentals": "1_analysts/fundamentals.md",
}


def _parse_report_id(report_id: str) -> tuple[str, str]:
    """Extract ticker and timestamp from report dir name like '小米集团_20260518_100034'."""
    m = re.match(r"^(.+)_(\d{8}_\d{6})$", report_id)
    if not m:
        raise ValueError(f"无效的报告 ID: {report_id}")
    ticker = m.group(1)
    ts = m.group(2)
    try:
        dt = datetime.strptime(ts, "%Y%m%d_%H%M%S")
        return ticker, dt.strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return ticker, ts


@app.get("/api/reports")
async def list_reports():
    """列出所有历史分析报告"""
    results = []
    if not REPORTS_DIR.exists():
        return JSONResponse(results)

    for d in sorted(REPORTS_DIR.iterdir(), reverse=True):
        if not d.is_dir():
            continue
        complete = d / "complete_report.md"
        if not complete.exists():
            continue
        try:
            ticker, date_str = _parse_report_id(d.name)
        except ValueError:
            continue

        # 提取决策（买入/持有/卖出）
        decision = ""
        text = complete.read_text(encoding="utf-8")
        tail = text[len(text)//2:]

        # 1) 显式评级标签: 评级：Underweight / 评级：买入
        m = re.search(r"评级[：:]\s*(Buy|Sell|Hold|Underweight|Overweight|买入|卖出|持有|减仓)", tail, re.IGNORECASE)
        if not m:
            # 2) 维持"买入"评级
            m = re.search(r'维持["\u300c]?\s*([买卖持][入有出])\s*["\u300d]?(?:评级)?', tail)
        if not m:
            # 3) 最终决策/方向性: 买入
            m = re.search(r"(?:最终(?:决策|评级|提案)|方向性|结论)[：:]\s*.*?([买卖持][入有出])", tail)
        if not m:
            # 4) 决定为/决定: 买入
            m = re.search(r"决定[了为]?[：:]*\s*([买卖持][入有出])", tail)
        if not m:
            # 5) 最终提案中包含减仓/建仓等方向词
            m_final = re.search(r"最终提案[：:]\s*(.+)", tail)
            if m_final:
                proposal = m_final.group(1)
                if re.search(r"减仓|减持|卖出|清仓|Underweight", proposal):
                    decision = "SELL"
                elif re.search(r"加仓|增持|建仓|买入|Buy|Overweight", proposal):
                    decision = "BUY"
                elif re.search(r"持有|Hold", proposal):
                    decision = "HOLD"

        if m and not decision:
            word = m.group(1).upper()
            decision = {
                "买入": "BUY", "卖出": "SELL", "持有": "HOLD",
                "BUY": "BUY", "SELL": "SELL", "HOLD": "HOLD",
                "OVERWEIGHT": "BUY", "UNDERWEIGHT": "SELL",
            }.get(word, word)

        results.append({
            "id": d.name,
            "ticker": ticker,
            "date": date_str,
            "decision": decision,
            "size": len(text),
        })

    return JSONResponse(results)


@app.get("/api/report/{report_id}")
async def get_report(report_id: str):
    """获取完整报告"""
    report_path = REPORTS_DIR / report_id / "complete_report.md"
    if not report_path.exists():
        return JSONResponse({"error": "报告不存在"}, status_code=404)
    return JSONResponse({
        "id": report_id,
        "content": report_path.read_text(encoding="utf-8"),
    })


@app.get("/api/report/{report_id}/section/{section}")
async def get_report_section(report_id: str, section: str):
    """获取报告中的某个子章节"""
    if section not in SECTION_FILES:
        return JSONResponse({"error": "无效的章节"}, status_code=400)

    file_path = REPORTS_DIR / report_id / SECTION_FILES[section]
    if not file_path.exists():
        return JSONResponse({"error": "章节不存在"}, status_code=404)

    return JSONResponse({
        "id": report_id,
        "section": section,
        "content": file_path.read_text(encoding="utf-8"),
    })
