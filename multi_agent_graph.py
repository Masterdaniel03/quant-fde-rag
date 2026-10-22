import asyncio
import re
import time
from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from llm_client import generate_draft
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


async def structured_quant_node(state: AgentState) -> dict:
    t0 = time.perf_counter()
    metrics = await asyncio.to_thread(
        engine.compute_structured_market_metrics,
        state["ticker"],
    )
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 3)
    traces = state.get("node_traces", []) + [
        {
            "node": "structured_numpy_quant",
            "latency_ms": elapsed_ms,
            "market_data_source": metrics["market_data_source"],
            "fallback_reason": metrics["fallback_reason"],
        }
    ]
    return {"quant_metrics": metrics, "node_traces": traces}


async def synthesis_and_self_reflection_node(state: AgentState) -> dict:
    t0 = time.perf_counter()
    m = state["quant_metrics"]
    citations = " ; ".join(state["compressed_citations"])

    deterministic_draft = (
        f"Executive Quantitative & Filing Synthesis for {m['ticker']}:\n"
        f"1. Unstructured SEC 10-K Evidence: "
        f"{citations if citations else 'No SEC 10-K passages were retrieved.'}\n"
        f"2. Structured Market Risk Metrics: Annualized Volatility is {m['annualized_volatility_pct']}%, "
        f"Sharpe Ratio is {m['sharpe_ratio']}, Max Drawdown is {m['max_drawdown_pct']}%, "
        f"and Latest Close is ${m['latest_price']}."
    )

    prompt = (
        "You are an institutional financial analyst. Synthesize the data concisely. "
        "Include the exact volatility and Sharpe values, and cite at least one SEC section "
        "using its supplied ticker and section label.\n"
        f"User question: {state['query']}\n"
        f"SEC 10-K evidence: {citations or 'No SEC 10-K passages were retrieved.'}\n"
        f"Quantitative ground truth: annualized volatility "
        f"{m['annualized_volatility_pct']}%, Sharpe ratio {m['sharpe_ratio']}, "
        f"max drawdown {m['max_drawdown_pct']}%, latest close ${m['latest_price']}."
    )
    generated_draft, usage, generation_error = await generate_draft(prompt)
    draft = generated_draft or deterministic_draft

    def passes_verification(candidate: str) -> bool:
        has_exact_volatility = str(m["annualized_volatility_pct"]) in candidate
        has_exact_sharpe = str(m["sharpe_ratio"]) in candidate
        cited_sections = [
            f"{doc['ticker']} | {doc['section']}"
            for doc in state["retrieved_docs"]
        ]
        has_citation = bool(cited_sections) and any(
            section in candidate for section in cited_sections
        )
        return has_exact_volatility and has_exact_sharpe and has_citation

    passed = passes_verification(draft)
    llm_status = "used" if generated_draft else "fallback"
    llm_fallback_reason = generation_error
    if generated_draft and not passed:
        draft = deterministic_draft
        passed = passes_verification(draft)
        llm_status = "fallback"
        llm_fallback_reason = "verification_failed"

    if passed:
        notes = "Verified exact numerical alignment with NumPy ground truth and SEC citations."
    elif not state["retrieved_docs"]:
        notes = "Numerical values were verified; no SEC filing passage was available to cite."
    else:
        notes = "Self-reflection detected a numerical discrepancy or missing SEC citation."

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 3)
    traces = state.get("node_traces", []) + [
        {
            "node": "synthesis_and_self_reflection",
            "latency_ms": elapsed_ms,
            "verified": passed,
            "llm_status": llm_status,
            "llm_fallback_reason": llm_fallback_reason,
        }
    ]

    total_latency_sec = sum(item["latency_ms"] for item in traces) / 1000.0
    input_tokens = usage.get("input_tokens", 0) if generated_draft else 0
    output_tokens = usage.get("output_tokens", 0) if generated_draft else 0
    if not generated_draft or not input_tokens:
        input_tokens = len(re.findall(r"\S+", prompt))
    if not generated_draft or not output_tokens:
        output_tokens = len(re.findall(r"\S+", draft))
    tokens_per_sec = round(output_tokens / max(total_latency_sec, 1e-4), 2)
    cost_per_request_usd = round((input_tokens * 0.00000015) + (output_tokens * 0.00000060), 8)

    telemetry = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_latency_ms": round(total_latency_sec * 1000, 3),
        "tokens_per_sec": tokens_per_sec,
        "cost_per_request_usd": cost_per_request_usd,
        "market_data_source": m["market_data_source"],
        "market_data_fallback_reason": m["fallback_reason"],
        "llm_status": llm_status,
        "llm_fallback_reason": llm_fallback_reason,
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
    result = asyncio.run(app_graph.ainvoke(initial_state))
    print(result["draft_answer"])
    print("Reflection Status:", result["reflection_passed"], "-", result["reflection_notes"])
    print("Granular Node Traces:", result["node_traces"])
    print("LLM Native Telemetry:", result["telemetry"])