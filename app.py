from fastapi import FastAPI
from pydantic import BaseModel
from multi_agent_graph import build_multi_agent_graph, AgentState

app = FastAPI(title="Quant SEC 10-K & Market Data Multi-Agent RAG Engine")
agent_compiled = build_multi_agent_graph()


class QueryRequest(BaseModel):
    query: str


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "quant-fde-multi-agent-rag"}


@app.post("/analyze")
def analyze_financial_query(req: QueryRequest):
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
    state_out = agent_compiled.invoke(initial_state)
    return {
        "ticker": state_out["ticker"],
        "answer": state_out["draft_answer"],
        "self_reflection_verified": state_out["reflection_passed"],
        "reflection_notes": state_out["reflection_notes"],
        "structured_quant_metrics": state_out["quant_metrics"],
        "unstructured_citations": state_out["compressed_citations"],
        "granular_traces": state_out["node_traces"],
        "llm_native_metrics": state_out["telemetry"],
    }