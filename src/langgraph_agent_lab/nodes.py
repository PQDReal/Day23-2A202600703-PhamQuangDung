"""Node functions for the LangGraph workflow.

Each function receives AgentState and returns a partial state update dict.
Do NOT mutate input state — return new values only.

LLM REQUIREMENT:
- classify_node MUST use a real LLM call (structured output for intent classification)
- answer_node MUST use a real LLM call (grounded response generation)
- evaluate_node SHOULD use LLM-as-judge (bonus points; heuristic acceptable for base score)
"""

from __future__ import annotations

import os
from typing import Literal

from pydantic import BaseModel, Field

from .llm import get_llm
from .state import AgentState, ApprovalDecision, make_event


class ClassificationResult(BaseModel):
    """Structured output contract for the classifier LLM."""

    route: Literal["simple", "tool", "missing_info", "risky", "error"] = Field(
        description="Best workflow route for the support request."
    )
    rationale: str = Field(description="Brief reason for the selected route.")


def _message_text(response: object) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content.strip()
    return str(content).strip()


def _fallback_classify(query: str) -> tuple[str, str]:
    """Rule fallback used only if the required LLM call fails."""
    text = query.lower()
    risky_terms = ("refund", "delete", "cancel", "send", "email", "chargeback", "remove")
    tool_terms = ("lookup", "order", "status", "tracking", "search", "find")
    error_terms = (
        "timeout",
        "failure",
        "failed",
        "crash",
        "unavailable",
        "cannot recover",
        "error",
    )
    vague_terms = ("fix it", "help", "issue", "problem", "not working")

    if any(term in text for term in risky_terms):
        return "risky", "fallback:risky keyword"
    if any(term in text for term in tool_terms):
        return "tool", "fallback:lookup keyword"
    if any(term in text for term in vague_terms) and len(text.split()) <= 5:
        return "missing_info", "fallback:vague request"
    if any(term in text for term in error_terms):
        return "error", "fallback:error keyword"
    return "simple", "fallback:general support question"


# ─── EXAMPLE: working node (provided for reference) ──────────────────
def intake_node(state: AgentState) -> dict:
    """Normalize raw query. This node is provided as a working example."""
    query = state.get("query", "").strip()
    return {
        "query": query,
        "messages": [f"intake:{query[:40]}"],
        "events": [make_event("intake", "completed", "query normalized")],
    }


# ─── Workflow nodes ─────────────────────────────────────────────────


def classify_node(state: AgentState) -> dict:
    """Classify the query into a route using an LLM.

    *** MUST use a real LLM call — keyword-only heuristics will lose points. ***

    Use .with_structured_output() or equivalent to get reliable enum classification.
    The LLM should classify into one of: simple, tool, missing_info, risky, error.

    Hints:
    - See llm.py for the get_llm() helper
    - Use Pydantic model or TypedDict with .with_structured_output()
    - Set risk_level to "high" for risky routes, "low" otherwise
    - Priority guide: risky > tool > missing_info > error > simple

    Return: {"route": str, "risk_level": str, "events": [make_event(...)]}
    """
    query = state.get("query", "")
    prompt = (
        "Classify this support ticket into exactly one route.\n"
        "Routes:\n"
        "- risky: side-effecting actions such as refunds, deletions, cancellations, "
        "sending emails.\n"
        "- tool: information lookups such as order status, tracking, account search.\n"
        "- missing_info: vague or incomplete requests lacking actionable detail.\n"
        "- error: system failures such as timeouts, crashes, service unavailable, "
        "unrecoverable failures.\n"
        "- simple: general support questions answerable without tools or side effects.\n"
        "Priority when multiple apply: risky > tool > missing_info > error > simple.\n"
        f"Ticket: {query}"
    )
    try:
        classifier = get_llm(temperature=0).with_structured_output(ClassificationResult)
        result = classifier.invoke(prompt)
        route = result.route
        rationale = result.rationale
        used_fallback = False
    except Exception as exc:
        route, rationale = _fallback_classify(query)
        used_fallback = True
        rationale = f"{rationale}; llm_error={type(exc).__name__}"

    return {
        "route": route,
        "risk_level": "high" if route == "risky" else "low",
        "messages": [f"classify:{route}"],
        "events": [
            make_event(
                "classify",
                "completed",
                f"classified as {route}",
                rationale=rationale,
                fallback=used_fallback,
            )
        ],
    }


def tool_node(state: AgentState) -> dict:
    """Execute a mock tool call.

    Simulate transient failures for error-route scenarios to test retry loops.

    Requirements:
    - Read current attempt count from state
    - If route is "error" and attempt < 2: return error result (string containing "ERROR")
    - Otherwise: return a mock success result string
    - Append result to tool_results list

    Return: {"tool_results": [result_string], "events": [make_event(...)]}
    """
    route = state.get("route", "")
    attempt = int(state.get("attempt", 0))
    query = state.get("query", "")
    approval = state.get("approval") or {}

    if route == "error" and attempt < 2:
        result = f"ERROR transient failure while processing attempt {attempt + 1}"
        event_type = "failed"
    elif route == "risky":
        approved_by = approval.get("reviewer", "unknown-reviewer")
        result = f"SUCCESS risky action prepared after approval by {approved_by}: {query}"
        event_type = "completed"
    elif route == "tool":
        result = f"SUCCESS lookup result for request: {query}"
        event_type = "completed"
    else:
        result = f"SUCCESS tool processed request: {query}"
        event_type = "completed"

    return {
        "tool_results": [result],
        "messages": [f"tool:{event_type}"],
        "events": [make_event("tool", event_type, result, attempt=attempt)],
    }


def evaluate_node(state: AgentState) -> dict:
    """Evaluate tool results — the retry-loop gate.

    Check whether the latest tool result is satisfactory or needs retry.

    SHOULD use LLM-as-judge for bonus points. Heuristic (e.g., check for "ERROR" substring)
    is acceptable for base score.

    Requirements:
    - Read the latest entry from tool_results
    - Set evaluation_result to "needs_retry" or "success"
    - This field drives route_after_evaluate conditional edge

    Note: You may need to add 'evaluation_result' to AgentState if not present.

    Return: {"evaluation_result": str, "events": [make_event(...)]}
    """
    latest_result = (state.get("tool_results") or [""])[-1]
    evaluation_result = "needs_retry" if "ERROR" in latest_result.upper() else "success"
    return {
        "evaluation_result": evaluation_result,
        "messages": [f"evaluate:{evaluation_result}"],
        "events": [
            make_event(
                "evaluate",
                "completed",
                f"tool result evaluation: {evaluation_result}",
                latest_result=latest_result,
            )
        ],
    }


def answer_node(state: AgentState) -> dict:
    """Generate a final response using an LLM.

    *** MUST use a real LLM call — hardcoded strings will lose points. ***

    The LLM should generate a helpful response grounded in available context:
    - tool_results (if any)
    - approval decision (if risky route)
    - original query

    Return: {"final_answer": str, "events": [make_event(...)]}
    """
    query = state.get("query", "")
    context = {
        "route": state.get("route", ""),
        "tool_results": state.get("tool_results", []),
        "approval": state.get("approval"),
        "errors": state.get("errors", []),
    }
    prompt = (
        "You are a concise support agent. Answer the user using only the provided context. "
        "Do not invent order details, refunds, or account changes. If the context says a tool "
        "succeeded, summarize the outcome; if no tool context exists, answer the general "
        "question.\n\n"
        f"User query: {query}\n"
        f"Context: {context}\n"
    )
    try:
        response = get_llm(temperature=0.2).invoke(prompt)
        answer = _message_text(response)
        used_fallback = False
    except Exception as exc:
        tool_results = state.get("tool_results") or []
        if tool_results:
            answer = f"Based on the available tool result: {tool_results[-1]}"
        else:
            answer = f"I can help with this request: {query}"
        used_fallback = True
        answer = f"{answer} (LLM fallback after {type(exc).__name__})"

    return {
        "final_answer": answer,
        "messages": ["answer:completed"],
        "events": [
            make_event("answer", "completed", "final answer generated", fallback=used_fallback)
        ],
    }


def ask_clarification_node(state: AgentState) -> dict:
    """Ask for missing information instead of hallucinating.

    Generate a specific clarification question based on the vague/incomplete query.

    Note: You may need to add 'pending_question' to AgentState if not present.

    Return: {"pending_question": str, "final_answer": str, "events": [make_event(...)]}
    """
    query = state.get("query", "")
    pending_question = (
        "Could you provide the specific account, order, or issue details needed to handle "
        f"this request: '{query}'?"
    )
    return {
        "pending_question": pending_question,
        "final_answer": pending_question,
        "messages": ["clarify:question"],
        "events": [make_event("clarify", "completed", "clarification requested")],
    }


def risky_action_node(state: AgentState) -> dict:
    """Prepare a risky action for human approval.

    Describe the proposed action and why it requires approval.

    Note: You may need to add 'proposed_action' to AgentState if not present.

    Return: {"proposed_action": str, "events": [make_event(...)]}
    """
    query = state.get("query", "")
    proposed_action = (
        "Human approval required before executing this side-effecting support action: "
        f"{query}"
    )
    return {
        "proposed_action": proposed_action,
        "risk_level": "high",
        "messages": ["risky_action:prepared"],
        "events": [make_event("risky_action", "completed", "risky action prepared")],
    }


def approval_node(state: AgentState) -> dict:
    """Human-in-the-loop approval step.

    Default behavior: mock approval (approved=True) so tests and CI run offline.
    Extension: if env LANGGRAPH_INTERRUPT=true, use langgraph.types.interrupt() for real HITL.

    Return:
        {"approval": {"approved": bool, "reviewer": str, "comment": str},
         "events": [make_event(...)]}
    """
    if os.getenv("LANGGRAPH_INTERRUPT", "").lower() == "true":
        from langgraph.types import interrupt

        payload = interrupt(
            {
                "proposed_action": state.get("proposed_action"),
                "query": state.get("query"),
                "instruction": "Approve or reject this risky action.",
            }
        )
        decision = ApprovalDecision.model_validate(payload)
    else:
        decision = ApprovalDecision(
            approved=True,
            reviewer="mock-reviewer",
            comment="Approved automatically for lab execution.",
        )

    return {
        "approval": decision.model_dump(),
        "messages": [f"approval:{decision.approved}"],
        "events": [
            make_event(
                "approval",
                "completed",
                "approval decision recorded",
                approved=decision.approved,
                reviewer=decision.reviewer,
            )
        ],
    }


def retry_or_fallback_node(state: AgentState) -> dict:
    """Record a retry attempt.

    Increment the attempt counter and log the transient failure.

    Requirements:
    - Read current attempt from state, increment by 1
    - Add an error message to errors list
    - Return updated attempt count

    Return: {"attempt": int, "errors": [str], "events": [make_event(...)]}
    """
    next_attempt = int(state.get("attempt", 0)) + 1
    error_message = f"retry attempt {next_attempt} after route={state.get('route', 'unknown')}"
    return {
        "attempt": next_attempt,
        "errors": [error_message],
        "messages": [f"retry:{next_attempt}"],
        "events": [
            make_event("retry", "completed", "retry attempt recorded", attempt=next_attempt)
        ],
    }


def dead_letter_node(state: AgentState) -> dict:
    """Handle unresolvable failures after max retries exceeded.

    This is the third layer: retry → fallback → dead letter.
    Log the failure and set a final_answer explaining that the request could not be completed.

    Return: {"final_answer": str, "events": [make_event(...)]}
    """
    answer = (
        "We could not complete the request after the maximum retry attempts. "
        "The case has been sent to the dead-letter queue for manual investigation."
    )
    return {
        "final_answer": answer,
        "messages": ["dead_letter:completed"],
        "events": [make_event("dead_letter", "completed", "max retries exhausted")],
    }


def finalize_node(state: AgentState) -> dict:
    """Emit a final audit event. All routes must pass through here before END.

    Return: {"events": [make_event("finalize", "completed", "workflow finished")]}
    """
    return {
        "messages": ["finalize:completed"],
        "events": [make_event("finalize", "completed", "workflow finished")],
    }
