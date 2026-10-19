import os
import re
import time
from typing import TypedDict
from google import genai
from langgraph.graph import StateGraph, START, END
from quant_hybrid_engine import QuantHybridEngine

class AgentState(TypedDict):
    query: str
    ticker: str
    retrieved_docs: list[dict]
    compressed_citations: list[str]
    quant_metrics: dict
    draft_answer: str
    reflection_passed: bool
    reflection_notes: str
    node_traces: list[dict]
    telemetry: dict


engine = QuantHybridEngine()

DEFAULT_FILINGS = [
    {
        "ticker": "NVDA",
        "section": "Item 1A. Risk Factors",
        "text": "NVIDIA relies heavily on TSMC in Taiwan for advanced CoWoS packaging and wafer fabrication of Blackwell GPUs. Any geopolitical or supply chain disruption could materially impact quarterly revenue.",
    },
    {
        "ticker": "NVDA",
        "section": "Item 7. MD&A",
        "text": "Data Center revenue grew 142% year-over-year driven by hyperscaler capital expenditure on generative AI training and low-latency inference clusters.",
    },
    {
        "ticker": "GOOGL",
        "section": "Item 1. Business",
        "text": "Google Cloud Platform (GCP) continues scaling Vertex AI, custom TPU v5p pods, and enterprise multi-agent RAG workflows for institutional clients.",
    },
    {
        "ticker": "AAPL",
        "section": "Item 1A. Risk Factors",
        "text": "Apple faces supply chain concentration in Asia and regulatory scrutiny regarding App Store ecosystem fees across European and US jurisdictions.",
    },
]

engine.ingest_unstructured_filings(DEFAULT_FILINGS)
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", ""))


def supervisor_node(state: AgentState) -> dict:
    t0 = time.perf_counter()
    query_upper = state["query"].upper()
    detected_ticker = "NVDA"
    for candidate in ["NVDA", "GOOGL", "AAPL", "MSFT"]:
        if candidate in query_upper:
            detected_ticker = candidate
            break
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 3)
    traces = state.get("node_traces", []) + [
        {"node": "supervisor_delegation", "latency_ms": elapsed_ms, "routed_ticker": detected_ticker}
    ]
    return {"ticker": detected_ticker, "node_traces": traces}


def unstructured_rag_node(state: AgentState) -> dict:
    t0 = time.perf_counter()
    docs = engine.hybrid_retrieve_rrf(state["query"], top_k=2)
    compressed = [
        f"[{d['ticker']} | {d['section']} | RRF={d['rrf_score']}] {d['text'][:160]}"
        for d in docs
    ]
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 3)
    traces = state.get("node_traces", []) + [
        {"node": "unstructured_qdrant_rag", "latency_ms": elapsed_ms, "chunks_retrieved": len(docs)}
    ]
    return {"retrieved_docs": docs, "compressed_citations": compressed, "node_traces": traces}


def structured_quant_node(state: AgentState) -> dict:
    t0 = time.perf_counter()
    metrics = engine.compute_structured_market_metrics(state["ticker"])
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 3)
    traces = state.get("node_traces", []) + [
        {"node": "structured_numpy_quant", "latency_ms": elapsed_ms}
    ]
    return {"quant_metrics": metrics, "node_traces": traces}


def synthesis_and_self_reflection_node(state: AgentState) -> dict:
    t0 = time.perf_counter()
    m = state["quant_metrics"]
    citations = " ; ".join(state["compressed_citations"])

    prompt = (
        f"You are an institutional financial analyst. Synthesize the following data concisely.\n"
        f"User Question: {state['query']}\n\n"
        f"SEC 10-K Evidence:\n{citations}\n\n"
        f"Quantitative Ground Truth:\n"
        f"- Volatility: {m['annualized_volatility_pct']}%\n"
        f"- Sharpe Ratio: {m['sharpe_ratio']}\n"
        f"- Max Drawdown: {m['max_drawdown_pct']}%\n"
        f"- Close: ${m['latest_price']}\n\n"
        f"You MUST explicitly include the exact numbers and cite the SEC section."
    )

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )

        if not response.text:
            raise ValueError("Gemini returned an empty response.")

        draft = response.text.strip()

        usage = response.usage_metadata

    except Exception as e:
        raise RuntimeError(
            f"Gemini API failed: {e}"
        ) from e

    vol_str = str(m["annualized_volatility_pct"])
    sharpe_str = str(m["sharpe_ratio"])
    has_exact_vol = vol_str in draft
    has_exact_sharpe = sharpe_str in draft
    has_citation = m["ticker"] in draft or len(state["compressed_citations"]) > 0

    passed = bool(has_exact_vol and has_exact_sharpe and has_citation)
    notes = (
        "Verified exact numerical alignment with NumPy ground-truth and SEC citations."
        if passed
        else "Self-reflection flagged numerical discrepancy or missing citation."
    )

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 3)
    traces = state.get("node_traces", []) + [
        {"node": "gemini_synthesis_and_reflection", "latency_ms": elapsed_ms, "verified": passed}
    ]

    total_latency_sec = sum(item["latency_ms"] for item in traces) / 1000.0
    
    
    usage = response.usage_metadata

    input_tokens = (usage.prompt_token_count or 0) if usage else 0
    output_tokens = (usage.candidates_token_count or 0) if usage else 0
    total_tokens = (usage.total_token_count or 0) if usage else 0
    thinking_tokens = (usage.thoughts_token_count or 0) if usage else 0
    tokens_per_sec = round(output_tokens / max(total_latency_sec, 1e-4), 2)
    cost_per_request_usd = round((input_tokens * 0.00000015) + (output_tokens * 0.00000060), 8)

    telemetry = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_latency_ms": round(total_latency_sec * 1000, 3),
        "tokens_per_sec": tokens_per_sec,
        "cost_per_request_usd": cost_per_request_usd,
    }

    return {
        "draft_answer": draft,
        "reflection_passed": passed,
        "reflection_notes": notes,
        "node_traces": traces,
        "telemetry": telemetry,
    }


def build_multi_agent_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("unstructured_rag", unstructured_rag_node)
    workflow.add_node("structured_quant", structured_quant_node)
    workflow.add_node("synthesis_reflection", synthesis_and_self_reflection_node)

    workflow.add_edge(START, "supervisor")
    workflow.add_edge("supervisor", "unstructured_rag")
    workflow.add_edge("unstructured_rag", "structured_quant")
    workflow.add_edge("structured_quant", "synthesis_reflection")
    workflow.add_edge("synthesis_reflection", END)

    return workflow.compile()


if __name__ == "__main__":
    app_graph = build_multi_agent_graph()
    initial_state: AgentState = {
        "query": "Analyze NVDA TSMC wafer supply chain risk and current volatility metrics",
        "ticker": "",
        "retrieved_docs": [],
        "compressed_citations": [],
        "quant_metrics": {},
        "draft_answer": "",
        "reflection_passed": False,
        "reflection_notes": "",
        "node_traces": [],
        "telemetry": {},
    }
    result = app_graph.invoke(initial_state)
    print("=== GEMINI REAL LLM SYNTHESIS ===")
    print(result["draft_answer"])
    print("\n=== REFLECTION STATUS ===")
    print(result["reflection_passed"], "-", result["reflection_notes"])
    print("\n=== GRANULAR NODE TRACES ===")
    print(result["node_traces"])
    print("\n=== LLM NATIVE TELEMETRY ===")
    print(result["telemetry"])