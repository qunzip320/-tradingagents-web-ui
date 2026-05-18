import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator

from dotenv import load_dotenv
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
from sse_starlette.sse import EventSourceResponse

load_dotenv(Path(__file__).parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).parent.parent))

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.graph.analyst_execution import build_analyst_execution_plan

app = FastAPI(title="TradingAgents Web")

STATIC_DIR = Path(__file__).parent / "static"

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
