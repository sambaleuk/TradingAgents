from __future__ import annotations

import argparse
import datetime as dt
import json
import mimetypes
import os
import threading
import time
import uuid
import webbrowser
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from socketserver import TCPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional at runtime
    load_dotenv = None

from cli.main import classify_message_type
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.checkpointer import clear_checkpoint, thread_id
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.llm_clients.model_catalog import MODEL_OPTIONS

if load_dotenv:
    load_dotenv()
    load_dotenv(".env.enterprise", override=False)


STATIC_DIR = Path(__file__).with_name("static")
RUNS: dict[str, "AnalysisRun"] = {}
RUNS_LOCK = threading.Lock()

ANALYST_ORDER = ["market", "social", "news", "fundamentals"]
ANALYST_REPORT_MAP = {
    "market": "market_report",
    "social": "sentiment_report",
    "news": "news_report",
    "fundamentals": "fundamentals_report",
}
AGENT_NAMES = {
    "market": "Market Analyst",
    "social": "Social Analyst",
    "news": "News Analyst",
    "fundamentals": "Fundamentals Analyst",
}
FIXED_AGENTS = [
    "Bull Researcher",
    "Bear Researcher",
    "Research Manager",
    "Trader",
    "Aggressive Analyst",
    "Neutral Analyst",
    "Conservative Analyst",
    "Portfolio Manager",
]
REPORT_TITLES = {
    "market_report": "Market Analysis",
    "sentiment_report": "Social Sentiment",
    "news_report": "News Analysis",
    "fundamentals_report": "Fundamentals",
    "investment_plan": "Research Decision",
    "trader_investment_plan": "Trader Plan",
    "final_trade_decision": "Portfolio Decision",
}

PROVIDER_KEY_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
    "xai": "XAI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "qwen": "DASHSCOPE_API_KEY",
    "glm": "ZHIPU_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}


class LocalThreadingHTTPServer(ThreadingHTTPServer):
    """HTTP server that avoids reverse DNS lookups during local startup."""

    def server_bind(self) -> None:
        TCPServer.server_bind(self)
        host, port = self.server_address[:2]
        self.server_name = host
        self.server_port = port


@dataclass
class AnalysisRun:
    run_id: str
    selections: dict[str, Any]
    status: str = "queued"
    error: str | None = None
    started_at: float = field(default_factory=time.time)
    completed_at: float | None = None
    agent_status: dict[str, str] = field(default_factory=dict)
    messages: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    reports: dict[str, str] = field(default_factory=dict)
    decision: str | None = None
    stats: dict[str, Any] = field(default_factory=dict)
    output_path: str | None = None
    _seen_message_ids: set[str] = field(default_factory=set, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def initialize(self) -> None:
        selected = self.selections["analysts"]
        self.agent_status = {
            **{AGENT_NAMES[a]: "pending" for a in selected if a in AGENT_NAMES},
            **{agent: "pending" for agent in FIXED_AGENTS},
        }
        self.reports = {
            key: ""
            for key, analyst in (
                ("market_report", "market"),
                ("sentiment_report", "social"),
                ("news_report", "news"),
                ("fundamentals_report", "fundamentals"),
            )
            if analyst in selected
        }
        self.reports.update(
            {
                "investment_plan": "",
                "trader_investment_plan": "",
                "final_trade_decision": "",
            }
        )

    def add_message(self, kind: str, content: str) -> None:
        if not content:
            return
        with self._lock:
            self.messages.append(
                {
                    "time": dt.datetime.now().strftime("%H:%M:%S"),
                    "type": kind,
                    "content": content[:4000],
                }
            )
            self.messages = self.messages[-150:]

    def add_tool_call(self, name: str, args: Any) -> None:
        with self._lock:
            self.tool_calls.append(
                {
                    "time": dt.datetime.now().strftime("%H:%M:%S"),
                    "name": name,
                    "args": args,
                }
            )
            self.tool_calls = self.tool_calls[-150:]

    def set_agent(self, agent: str, status: str) -> None:
        with self._lock:
            if agent in self.agent_status:
                self.agent_status[agent] = status

    def set_report(self, key: str, value: str) -> None:
        if key in self.reports and value:
            with self._lock:
                self.reports[key] = value

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            total_agents = len(self.agent_status)
            completed = sum(1 for s in self.agent_status.values() if s == "completed")
            elapsed_to = self.completed_at or time.time()
            return {
                "run_id": self.run_id,
                "status": self.status,
                "error": self.error,
                "selections": self.selections,
                "agent_status": self.agent_status,
                "messages": self.messages,
                "tool_calls": self.tool_calls,
                "reports": self.reports,
                "report_titles": REPORT_TITLES,
                "decision": self.decision,
                "stats": {
                    **self.stats,
                    "elapsed_seconds": int(elapsed_to - self.started_at),
                    "agents_completed": completed,
                    "agents_total": total_agents,
                    "reports_completed": sum(1 for v in self.reports.values() if v),
                    "reports_total": len(self.reports),
                },
                "output_path": self.output_path,
            }


def preferred_provider() -> str:
    default_provider = str(DEFAULT_CONFIG["llm_provider"]).lower()
    if os.getenv(PROVIDER_KEY_ENV.get(default_provider, "")):
        return default_provider
    for provider, env_var in PROVIDER_KEY_ENV.items():
        if os.getenv(env_var):
            return provider
    return default_provider


def default_model(provider: str, mode: str) -> str:
    options = MODEL_OPTIONS.get(provider, {}).get(mode, [])
    if options:
        return options[0][1]
    return DEFAULT_CONFIG["quick_think_llm" if mode == "quick" else "deep_think_llm"]


def normalize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    analysts = payload.get("analysts") or ANALYST_ORDER
    analysts = [a for a in analysts if a in ANALYST_ORDER]
    if not analysts:
        analysts = ["market", "news"]

    today = dt.datetime.now().strftime("%Y-%m-%d")
    analysis_date = str(payload.get("analysis_date") or today)
    dt.datetime.strptime(analysis_date, "%Y-%m-%d")
    if dt.datetime.strptime(analysis_date, "%Y-%m-%d").date() > dt.date.today():
        raise ValueError("Analysis date cannot be in the future.")

    provider = str(payload.get("llm_provider") or preferred_provider()).lower()
    if provider not in MODEL_OPTIONS and provider not in {"openrouter", "azure"}:
        raise ValueError(f"Unsupported LLM provider: {provider}")

    return {
        "ticker": str(payload.get("ticker") or "SPY").strip().upper(),
        "analysis_date": analysis_date,
        "analysts": analysts,
        "research_depth": int(payload.get("research_depth") or 1),
        "llm_provider": provider,
        "backend_url": payload.get("backend_url") or DEFAULT_CONFIG.get("backend_url"),
        "quick_think_llm": str(payload.get("quick_think_llm") or default_model(provider, "quick")),
        "deep_think_llm": str(payload.get("deep_think_llm") or default_model(provider, "deep")),
        "output_language": str(payload.get("output_language") or DEFAULT_CONFIG["output_language"]),
        "checkpoint_enabled": bool(payload.get("checkpoint_enabled", False)),
        "google_thinking_level": payload.get("google_thinking_level") or None,
        "openai_reasoning_effort": payload.get("openai_reasoning_effort") or None,
        "anthropic_effort": payload.get("anthropic_effort") or None,
    }


def build_config(selections: dict[str, Any]) -> dict[str, Any]:
    config = DEFAULT_CONFIG.copy()
    config["max_debate_rounds"] = selections["research_depth"]
    config["max_risk_discuss_rounds"] = selections["research_depth"]
    config["llm_provider"] = selections["llm_provider"]
    config["quick_think_llm"] = selections["quick_think_llm"]
    config["deep_think_llm"] = selections["deep_think_llm"]
    config["backend_url"] = selections["backend_url"]
    config["output_language"] = selections["output_language"]
    config["checkpoint_enabled"] = selections["checkpoint_enabled"]
    config["google_thinking_level"] = selections["google_thinking_level"]
    config["openai_reasoning_effort"] = selections["openai_reasoning_effort"]
    config["anthropic_effort"] = selections["anthropic_effort"]
    return config


def run_analysis(run: AnalysisRun) -> None:
    run.initialize()
    run.status = "running"
    selections = run.selections
    selected_analysts = [a for a in ANALYST_ORDER if a in selections["analysts"]]
    run.add_message("System", f"Starting {selections['ticker']} analysis for {selections['analysis_date']}")

    config = build_config(selections)
    graph = None
    checkpointer_ctx = None

    try:
        graph = TradingAgentsGraph(
            selected_analysts=selected_analysts,
            debug=False,
            config=config,
        )
        graph.ticker = selections["ticker"]
        graph._resolve_pending_entries(selections["ticker"])

        if selected_analysts:
            run.set_agent(AGENT_NAMES[selected_analysts[0]], "in_progress")

        if config.get("checkpoint_enabled"):
            from tradingagents.graph.checkpointer import get_checkpointer

            checkpointer_ctx = get_checkpointer(config["data_cache_dir"], selections["ticker"])
            saver = checkpointer_ctx.__enter__()
            graph.graph = graph.workflow.compile(checkpointer=saver)

        initial_state = graph.propagator.create_initial_state(
            selections["ticker"],
            selections["analysis_date"],
            past_context=graph.memory_log.get_past_context(selections["ticker"]),
        )
        args = graph.propagator.get_graph_args()
        if config.get("checkpoint_enabled"):
            args.setdefault("config", {}).setdefault("configurable", {})["thread_id"] = thread_id(
                selections["ticker"], selections["analysis_date"]
            )

        final_state = None
        for chunk in graph.graph.stream(initial_state, **args):
            final_state = chunk
            handle_chunk(run, chunk)

        if not final_state:
            raise RuntimeError("TradingAgents returned no final state.")

        graph.curr_state = final_state
        graph._log_state(selections["analysis_date"], final_state)
        graph.memory_log.store_decision(
            ticker=selections["ticker"],
            trade_date=selections["analysis_date"],
            final_trade_decision=final_state["final_trade_decision"],
        )
        if config.get("checkpoint_enabled"):
            clear_checkpoint(config["data_cache_dir"], selections["ticker"], selections["analysis_date"])

        for key in run.reports:
            if key in final_state and final_state[key]:
                run.set_report(key, final_state[key])
        run.decision = graph.process_signal(final_state["final_trade_decision"])
        with run._lock:
            for agent in run.agent_status:
                run.agent_status[agent] = "completed"
            run.output_path = str(Path(config["results_dir"]) / selections["ticker"])
            run.status = "completed"
            run.completed_at = time.time()
        run.add_message("System", "Analysis completed")
    except Exception as exc:  # pragma: no cover - exercised manually with real APIs
        with run._lock:
            run.status = "failed"
            run.error = str(exc)
            run.completed_at = time.time()
        run.add_message("Error", str(exc))
    finally:
        if checkpointer_ctx is not None:
            checkpointer_ctx.__exit__(None, None, None)


def handle_chunk(run: AnalysisRun, chunk: dict[str, Any]) -> None:
    for message in chunk.get("messages", []):
        message_id = getattr(message, "id", None)
        if message_id and message_id in run._seen_message_ids:
            continue
        if message_id:
            run._seen_message_ids.add(message_id)

        msg_type, content = classify_message_type(message)
        if content:
            run.add_message(msg_type, content)

        for tool_call in getattr(message, "tool_calls", []) or []:
            if isinstance(tool_call, dict):
                run.add_tool_call(tool_call.get("name", "tool"), tool_call.get("args", {}))
            else:
                run.add_tool_call(tool_call.name, tool_call.args)

    update_analysts(run, chunk)
    update_research(run, chunk)
    update_trader(run, chunk)
    update_risk(run, chunk)


def update_analysts(run: AnalysisRun, chunk: dict[str, Any]) -> None:
    selected = run.selections["analysts"]
    active_set = False
    for analyst in ANALYST_ORDER:
        if analyst not in selected:
            continue
        report_key = ANALYST_REPORT_MAP[analyst]
        agent_name = AGENT_NAMES[analyst]
        if chunk.get(report_key):
            run.set_report(report_key, chunk[report_key])
        has_report = bool(run.snapshot()["reports"].get(report_key))
        if has_report:
            run.set_agent(agent_name, "completed")
        elif not active_set:
            run.set_agent(agent_name, "in_progress")
            active_set = True
        else:
            run.set_agent(agent_name, "pending")

    if not active_set and selected:
        run.set_agent("Bull Researcher", "in_progress")


def update_research(run: AnalysisRun, chunk: dict[str, Any]) -> None:
    debate = chunk.get("investment_debate_state") or {}
    bull = (debate.get("bull_history") or "").strip()
    bear = (debate.get("bear_history") or "").strip()
    judge = (debate.get("judge_decision") or "").strip()
    if bull or bear:
        for agent in ("Bull Researcher", "Bear Researcher", "Research Manager"):
            run.set_agent(agent, "in_progress")
    if bull:
        run.set_report("investment_plan", f"### Bull Researcher\n{bull}")
    if bear:
        run.set_report("investment_plan", f"### Bear Researcher\n{bear}")
    if judge:
        run.set_report("investment_plan", f"### Research Manager\n{judge}")
        for agent in ("Bull Researcher", "Bear Researcher", "Research Manager"):
            run.set_agent(agent, "completed")
        run.set_agent("Trader", "in_progress")


def update_trader(run: AnalysisRun, chunk: dict[str, Any]) -> None:
    plan = chunk.get("trader_investment_plan")
    if plan:
        run.set_report("trader_investment_plan", plan)
        run.set_agent("Trader", "completed")
        run.set_agent("Aggressive Analyst", "in_progress")


def update_risk(run: AnalysisRun, chunk: dict[str, Any]) -> None:
    risk = chunk.get("risk_debate_state") or {}
    pieces = []
    for key, agent in (
        ("aggressive_history", "Aggressive Analyst"),
        ("neutral_history", "Neutral Analyst"),
        ("conservative_history", "Conservative Analyst"),
    ):
        text = (risk.get(key) or "").strip()
        if text:
            run.set_agent(agent, "in_progress")
            pieces.append(f"### {agent}\n{text}")
    judge = (risk.get("judge_decision") or "").strip()
    if judge:
        pieces.append(f"### Portfolio Manager\n{judge}")
        for agent in (
            "Aggressive Analyst",
            "Neutral Analyst",
            "Conservative Analyst",
            "Portfolio Manager",
        ):
            run.set_agent(agent, "completed")
    if pieces:
        run.set_report("final_trade_decision", "\n\n".join(pieces))


def market_snapshot(ticker: str) -> dict[str, Any]:
    try:
        import yfinance as yf

        ticker_obj = yf.Ticker(ticker)
        history = ticker_obj.history(period="1mo")
        if history.empty:
            return {"ticker": ticker, "error": "No market data returned."}
        close = history["Close"]
        last = float(close.iloc[-1])
        prev = float(close.iloc[-2]) if len(close) > 1 else last
        return {
            "ticker": ticker,
            "last": round(last, 2),
            "change": round(last - prev, 2),
            "change_pct": round(((last - prev) / prev) * 100, 2) if prev else 0,
            "sparkline": [round(float(v), 2) for v in close.tail(24).tolist()],
        }
    except Exception as exc:  # pragma: no cover - network dependent
        return {"ticker": ticker, "error": str(exc)}


class TradingAgentsGUIHandler(BaseHTTPRequestHandler):
    server_version = "TradingAgentsGUI/0.1"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.serve_static("index.html")
        elif parsed.path == "/favicon.ico":
            self.send_response(HTTPStatus.NO_CONTENT)
            self.end_headers()
        elif parsed.path.startswith("/static/"):
            self.serve_static(parsed.path.removeprefix("/static/"))
        elif parsed.path == "/api/config":
            provider = preferred_provider()
            self.write_json(
                {
                    "defaults": {
                        "ticker": "SPY",
                        "analysis_date": dt.datetime.now().strftime("%Y-%m-%d"),
                        "analysts": ANALYST_ORDER,
                        "research_depth": 1,
                        "llm_provider": provider,
                        "quick_think_llm": default_model(provider, "quick"),
                        "deep_think_llm": default_model(provider, "deep"),
                        "output_language": DEFAULT_CONFIG["output_language"],
                    },
                    "models": MODEL_OPTIONS,
                }
            )
        elif parsed.path == "/api/run":
            run_id = parse_qs(parsed.query).get("id", [""])[0]
            with RUNS_LOCK:
                run = RUNS.get(run_id)
            if not run:
                self.write_json({"error": "Run not found"}, HTTPStatus.NOT_FOUND)
                return
            self.write_json(run.snapshot())
        elif parsed.path == "/api/market":
            ticker = parse_qs(parsed.query).get("ticker", ["SPY"])[0].upper()
            self.write_json(market_snapshot(ticker))
        else:
            self.write_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/run":
            self.write_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length).decode("utf-8")
            selections = normalize_payload(json.loads(raw or "{}"))
        except Exception as exc:
            self.write_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return

        run_id = uuid.uuid4().hex
        run = AnalysisRun(run_id=run_id, selections=selections)
        with RUNS_LOCK:
            RUNS[run_id] = run
        thread = threading.Thread(target=run_analysis, args=(run,), daemon=True)
        thread.start()
        self.write_json({"run_id": run_id}, HTTPStatus.CREATED)

    def serve_static(self, name: str) -> None:
        path = (STATIC_DIR / name).resolve()
        if not str(path).startswith(str(STATIC_DIR.resolve())) or not path.is_file():
            self.write_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
            return
        content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        data = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def write_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: Any) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the TradingAgents graphical interface.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    server = LocalThreadingHTTPServer((args.host, args.port), TradingAgentsGUIHandler)
    url = f"http://{args.host}:{args.port}"
    print(f"TradingAgents GUI running at {url}")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping TradingAgents GUI")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
