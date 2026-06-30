# Day 08 Lab Report

## 1. Team / student

- Name: Pham Quang Dung
- Repo/commit: local workspace
- Date: 2026-06-30

## 2. Architecture

The workflow is a LangGraph `StateGraph` for support-ticket orchestration. The fixed entry
path is `START -> intake -> classify`. Classification chooses one of five routes:
`simple`, `tool`, `missing_info`, `risky`, or `error`.

The route branches are:

- `simple -> answer -> finalize -> END`
- `tool -> tool -> evaluate -> answer/retry`
- `missing_info -> clarify -> finalize -> END`
- `risky -> risky_action -> approval -> tool/clarify`
- `error -> retry -> tool/dead_letter`

All successful, clarification, and dead-letter paths converge at `finalize`, which records the
last audit event before termination.

LLM integration is used where the rubric requires it: `classify_node` calls
`.with_structured_output()` with a typed classification schema, and `answer_node` generates a
grounded final response from the query, tool results, approval decision, and errors. The
`evaluate_node` uses a deterministic error check for the base retry gate.

## 3. State schema

| Field | Reducer | Why |
|---|---|---|
| `thread_id` | overwrite | Stable LangGraph checkpoint key per scenario run. |
| `scenario_id` | overwrite | Connects final state back to scenario metrics. |
| `query` | overwrite | Normalized support ticket text. |
| `route` | overwrite | Current classified route. |
| `risk_level` | overwrite | Marks high-risk requests for approval handling. |
| `attempt` | overwrite | Current bounded retry count. |
| `max_attempts` | overwrite | Scenario-specific retry ceiling. |
| `evaluation_result` | overwrite | Retry gate after tool evaluation. |
| `pending_question` | overwrite | Clarification output for vague or rejected requests. |
| `proposed_action` | overwrite | Risky action description for approval. |
| `approval` | overwrite | Mock or real HITL approval decision. |
| `final_answer` | overwrite | Final user-visible response. |
| `messages` | append | Lightweight execution trace. |
| `tool_results` | append | Keeps each tool attempt visible for retry analysis. |
| `errors` | append | Records retry and failure context. |
| `events` | append | Primary audit log used by metrics. |

## 4. Metrics summary

| Metric | Value |
|---|---:|
| Total scenarios | 7 |
| Success rate | 100.00% |
| Average nodes visited | 6.43 |
| Total retries | 3 |
| Total approval/HITL events | 2 |
| Resume success demonstrated | yes |

## 5. Scenario results

| Scenario | Expected route | Actual route | Success | Retries | Interrupts | Approval required | Approval observed |
|---|---|---|---|---:|---:|---|---|
| S01_simple | simple | simple | yes | 0 | 0 | no | no |
| S02_tool | tool | tool | yes | 0 | 0 | no | no |
| S03_missing | missing_info | missing_info | yes | 0 | 0 | no | no |
| S04_risky | risky | risky | yes | 0 | 1 | yes | yes |
| S05_error | error | error | yes | 2 | 0 | no | no |
| S06_delete | risky | risky | yes | 0 | 1 | yes | yes |
| S07_dead_letter | error | error | yes | 1 | 0 | no | no |

## 6. Failure analysis

1. Retry or tool failure: tool errors are represented as tool results containing `ERROR`.
   `evaluate` converts those into `evaluation_result = needs_retry`, and `retry` increments
   `attempt`. `route_after_retry` sends the workflow back to `tool` only while
   `attempt < max_attempts`; otherwise it sends the case to `dead_letter`.

2. Risky action without approval: side-effecting requests such as refunds, deletions, and
   confirmation emails route to `risky_action` first. The workflow cannot reach `tool` from
   that branch until `approval` records an approved decision. Rejections route to
   `clarify` instead.

3. Missing information: vague requests route to `clarify`, which sets both
   `pending_question` and `final_answer` so the run terminates cleanly without hallucinating
   details.

Current failed scenarios:

- No scenario failures in the current metrics run.

## 7. Persistence / recovery evidence

The CLI invokes the graph with `configurable.thread_id` from the scenario state, so every run
has a stable checkpoint identity such as `thread-S01_simple`. The default config uses the
in-memory checkpointer for quick grading runs. The persistence extension is implemented in
`build_checkpointer("sqlite", database_url)`, which creates a SQLite-backed `SqliteSaver`,
enables WAL mode, and stores checkpoints under `outputs/checkpoints.sqlite` unless another
path is provided. Local evidence is available in `outputs/persistence_evidence.txt`; it records
the SQLite thread id, final route, final-answer presence, finalize observation, and state-history
snapshot count.

## 8. Extension work

- SQLite checkpointer support is implemented for durable checkpoints.
- Mock HITL approval is implemented by default; setting `LANGGRAPH_INTERRUPT=true` switches
  the approval node to LangGraph interrupt-based review.
- Events are emitted at each node so the metrics file can count routes, retries, approval
  events, and finalization.
- A Mermaid graph diagram is exported to `outputs/graph.mmd` with
  `graph.get_graph().draw_mermaid()`.

## 9. Improvement plan

With one more day, I would replace the mock support tool with typed external tool calls, add
LLM-as-judge evaluation for tool result quality, build a small operator UI for real
interrupt/resume approval, and automate crash-recovery replay as part of CI.
