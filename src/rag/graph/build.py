"""Build and compilation of the LangGraph state machine."""

from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from rag.graph.nodes import (
    finalize_node,
    generate_node,
    grade_node,
    query_prep_node,
    retrieve_node,
    verify_node,
)
from rag.graph.routing import (
    route_after_generate,
    route_after_grade,
    route_after_retrieve,
    route_after_verify,
)
from rag.graph.state import RagState


def build_graph(
    checkpointer: BaseCheckpointSaver[Any] | None = None,
) -> Any:
    """Compile the LangGraph state machine.

    Args:
        checkpointer: Checkpoint saver for state persistence (defaults to in-memory).

    Returns:
        Compiled LangGraph runnable.
    """
    workflow = StateGraph(RagState)

    # Register nodes
    workflow.add_node("query_prep", query_prep_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade", grade_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("verify", verify_node)
    workflow.add_node("finalize", finalize_node)

    # Linear and conditional edges
    workflow.add_edge(START, "query_prep")
    workflow.add_edge("query_prep", "retrieve")

    workflow.add_conditional_edges(
        "retrieve",
        route_after_retrieve,
        {
            "grade": "grade",
            "finalize": "finalize",
        },
    )

    workflow.add_conditional_edges(
        "grade",
        route_after_grade,
        {
            "generate": "generate",
            "finalize": "finalize",
        },
    )

    workflow.add_conditional_edges(
        "generate",
        route_after_generate,
        {
            "verify": "verify",
            "finalize": "finalize",
        },
    )

    workflow.add_conditional_edges(
        "verify",
        route_after_verify,
        {
            "finalize": "finalize",
        },
    )

    workflow.add_edge("finalize", END)

    saver = checkpointer if checkpointer is not None else MemorySaver()
    return workflow.compile(checkpointer=saver)
