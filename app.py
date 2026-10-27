"""Streamlit interactive application for Resilient RAG."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from rag.bootstrap import Components, build_components
from rag.chaos.controller import FaultController
from rag.chaos.loader import load_scenarios
from rag.chaos.wrappers import (
    ChaosBM25Index,
    ChaosChatModel,
    ChaosEmbedder,
    ChaosRetriever,
    ChaosVectorStore,
)
from rag.clock import VirtualClock
from rag.config import get_settings
from rag.schemas import Outcome


@st.cache_resource
def get_components() -> Components:
    """Build and cache the base pipeline components."""
    settings = get_settings()
    return build_components(settings)


def get_available_scenarios() -> list[str]:
    """Load scenario names from scenarios YAML."""
    path = Path("configs/scenarios.yaml")
    if not path.is_file():
        return ["None"]
    try:
        scenarios = load_scenarios(path)
        return ["None"] + [s.name for s in scenarios]
    except Exception:
        return ["None"]


def render_ui() -> None:
    """Render the full Streamlit UI."""
    st.set_page_config(
        page_title="Resilient RAG Demo",
        page_icon="🛡️",
        layout="wide",
    )

    st.title("🛡️ Resilient RAG: Self-Healing Pipeline Demo")
    st.markdown(
        "A fault-tolerant Retrieval-Augmented Generation pipeline with **deterministic self-healing**, "
        "circuit breakers, and chaos injection."
    )

    # Sidebar configuration
    st.sidebar.header("Pipeline Controls")
    healing_enabled = st.sidebar.checkbox(
        "Enable Self-Healing Supervisor",
        value=True,
        help="When enabled, detects failures and executes bounded recovery actions.",
    )

    st.sidebar.subheader("Chaos Injection")
    available_scenarios = get_available_scenarios()
    selected_scenario = st.sidebar.selectbox(
        "Inject Chaos Scenario",
        options=available_scenarios,
        index=0,
        help="Simulate real-world infrastructure and model failures.",
    )

    fault_seed = st.sidebar.number_input("Chaos Seed", value=42, min_value=1, step=1)
    retrieval_mode = st.sidebar.selectbox(
        "Retrieval Mode", options=["hybrid", "vector", "bm25"], index=0
    )
    top_k = st.sidebar.slider("Top K Chunks", min_value=1, max_value=10, value=5)

    # Question inputs
    st.subheader("Query")
    sample_queries = [
        "What is the primary role of the kubelet on worker nodes?",
        "How do readiness and liveness probes differ in Kubernetes?",
        "What are the three Pod Security Standards levels?",
        "How does the Horizontal Pod Autoscaler scale workloads?",
    ]

    selected_sample = st.selectbox(
        "Or choose a benchmark sample query:", ["(Custom query)"] + sample_queries
    )
    if selected_sample != "(Custom query)":
        default_query = selected_sample
    else:
        default_query = "What is the primary role of the kubelet on worker nodes?"

    query = st.text_input("Enter your question:", value=default_query)

    if st.button("Submit Query", type="primary"):
        if not query.strip():
            st.warning("Please enter a valid query.")
            return

        with st.spinner("Executing Resilient RAG pipeline..."):
            components = get_components()

            # Optional chaos wrapping
            if selected_scenario != "None":
                scenarios_path = Path("configs/scenarios.yaml")
                scenarios = [
                    s for s in load_scenarios(scenarios_path) if s.name == selected_scenario
                ]
                if scenarios:
                    clock = VirtualClock(start_time=0.0)
                    controller = FaultController(faults=scenarios[0].faults, seed=int(fault_seed))

                    components.embedder = ChaosEmbedder(
                        target=components.embedder,
                        controller=controller,
                        clock=clock,
                    )
                    components.vector_store = ChaosVectorStore(
                        target=components.vector_store,
                        controller=controller,
                        clock=clock,
                    )
                    components.bm25_index = ChaosBM25Index(
                        target=components.bm25_index,
                        controller=controller,
                        clock=clock,
                    )
                    components.retriever = ChaosRetriever(
                        target=components.retriever,
                        controller=controller,
                        clock=clock,
                    )
                    active_model = components.llm_client.tiers[0]
                    chaos_model = ChaosChatModel(
                        target=active_model,
                        controller=controller,
                        clock=clock,
                    )
                    components.llm_client.tiers[0] = chaos_model
                    components.llm_client.clock = clock

            # Execute pipeline
            result_state = components.run(
                question=query,
                healing_enabled=healing_enabled,
                retrieval_mode=retrieval_mode,
                top_k=top_k,
            )

        # Render Outcome
        raw_outcome = result_state.get("outcome")
        outcome_str = (
            raw_outcome.value if isinstance(raw_outcome, Outcome) else str(raw_outcome or "failed")
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            if outcome_str == Outcome.ANSWERED.value:
                st.success(f"Outcome: **{outcome_str}**")
            elif outcome_str == Outcome.ABSTAINED.value:
                st.warning(f"Outcome: **{outcome_str}** (Safe Abstention)")
            elif outcome_str == Outcome.ESCALATED.value:
                st.error(f"Outcome: **{outcome_str}** (Human Escalation)")
            else:
                st.error(f"Outcome: **{outcome_str}**")
        with col2:
            st.metric("LLM Invocations", result_state.get("llm_calls", 0))
        with col3:
            st.metric("Total Tokens", result_state.get("token_usage", 0))

        # Render Answer
        st.subheader("Answer")
        answer_text = result_state.get("answer", "")
        if answer_text:
            st.markdown(answer_text)
        else:
            st.info("No answer generated.")

        # Citations
        citations = result_state.get("citations", [])
        if citations:
            with st.expander(f"📚 Citations ({len(citations)})", expanded=True):
                for c in citations:
                    st.markdown(f"**Chunk ID:** `{c.chunk_id}`")
                    st.caption(c.text)

        # Healing Supervisor Report
        heal_attempts = result_state.get("heal_attempts", 0)
        heal_history = result_state.get("heal_history", [])
        quarantined = result_state.get("quarantined_ids", [])

        with st.expander("🩺 Self-Healing Supervisor Report", expanded=heal_attempts > 0):
            st.write(f"**Total Recovery Attempts:** {heal_attempts}")
            if heal_history:
                st.write("**Recovery Sequence:**")
                for idx, (fail, act) in enumerate(heal_history, 1):
                    st.write(f"{idx}. Detected Failure: `{fail}` ➔ Recovery Action: `{act}`")
            else:
                st.write("No faults triggered recovery supervisor.")

            if quarantined:
                st.write(f"**Quarantined Chunks:** {quarantined}")

        # Pipeline Trace Timeline
        trace = result_state.get("trace", [])
        if trace:
            with st.expander("⏱️ Pipeline Execution Timeline", expanded=False):
                trace_data: list[dict[str, Any]] = [
                    {
                        "Node": t.node,
                        "Timestamp": round(t.timestamp, 4),
                        "Details": str(t.details) if t.details else "",
                    }
                    for t in trace
                ]
                st.table(trace_data)


def main() -> None:
    render_ui()


if __name__ == "__main__":
    main()
