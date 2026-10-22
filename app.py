from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from multi_agent_graph import build_multi_agent_graph, AgentState

app = FastAPI(title="Quant SEC 10-K & Market Data Multi-Agent RAG Engine")
agent_compiled = build_multi_agent_graph()


class QueryRequest(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", str_strip_whitespace=True)

    query: str = Field(min_length=1, max_length=2000)


class QuantMetrics(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    ticker: str
    annualized_volatility_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    latest_price: float
    market_data_source: Literal["yfinance", "deterministic_fallback"]
    fallback_reason: str | None


class NodeTrace(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    node: str
    latency_ms: float
    routed_ticker: str | None = None
    chunks_retrieved: int | None = None
    market_data_source: Literal["yfinance", "deterministic_fallback"] | None = None
    fallback_reason: str | None = None
    verified: bool | None = None
    llm_status: Literal["used", "fallback"] | None = None
    llm_fallback_reason: str | None = None


class Telemetry(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    input_tokens: int
    output_tokens: int
    total_latency_ms: float
    tokens_per_sec: float
    cost_per_request_usd: float
    market_data_source: Literal["yfinance", "deterministic_fallback"]
    market_data_fallback_reason: str | None
    llm_status: Literal["used", "fallback"]
    llm_fallback_reason: str | None


class AnalyzeResponse(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    ticker: str
    draft_answer: str
    quant_metrics: QuantMetrics
    reflection_passed: bool
    reflection_notes: str
    node_traces: list[NodeTrace]
    telemetry: Telemetry
    citations: list[str]


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "quant-fde-multi-agent-rag"}


@app.post("/analyze")
async def analyze_financial_query(req: QueryRequest) -> AnalyzeResponse:
    initial_state: AgentState = {
        "query": req.query,
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
    state_out = await agent_compiled.ainvoke(initial_state)
    return AnalyzeResponse(
        ticker=state_out["ticker"],
        draft_answer=state_out["draft_answer"],
        quant_metrics=state_out["quant_metrics"],
        reflection_passed=state_out["reflection_passed"],
        reflection_notes=state_out["reflection_notes"],
        node_traces=state_out["node_traces"],
        telemetry=state_out["telemetry"],
        citations=state_out["compressed_citations"],
    )