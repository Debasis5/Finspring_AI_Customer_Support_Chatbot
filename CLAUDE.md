# CLAUDE.md — RAG-Based Agentic Chatbot (v1)

Project instructions for AI-assisted development. **Scope is v1 only** (Architecture 2 — LangGraph state machine). Do not implement v2/Architecture 3 features (agentic subgraphs, text-to-SQL, supervisor, computation tool) — those are a documented future migration, not current scope.

The full design rationale lives in `docs/architecture_v1_reference.md`. This file is the operational summary: what to build, the invariants to never break, and the conventions to follow.

---

## 1. What this project is

A customer-facing chatbot for a B2B fintech investment-infrastructure platform. It answers exactly two classes of questions:

1. **Product documentation Q&A** — Digital Fixed Deposits, Digital Gold, Bonds, Mutual Funds — answered **strictly from ingested product documentation**.
2. **Customer-specific account queries** — holdings, accounts, transactions — answered from PostgreSQL (1 demographics table + 4 product tables).

Everything else is declined or, if in-scope-but-unanswerable, routed to a human via a support ticket with the customer's consent.

**Stack (fixed):** Python · LangChain · LangGraph · OpenAI API · FastAPI · Streamlit (demo UI) · PostgreSQL + pgvector (vector store — deliberately no new infra).

---

## 2. Architecture — the graph (v1, final)

One LangGraph `StateGraph`, checkpointed with `PostgresSaver`. Topology:

```
__start__ → pre_checks → classify
  pre_checks --blocked--> decline
  classify --off_topic|unsafe|prohibited_advice--> decline → __end__
  classify --docs--> retrieve_docs ─┐
  classify --account--> retrieve_db ┤ (parallel branches when "both")
  classify --both--> both nodes ────┘
        → generate → verify
  verify --grounded ∧ no advice--> respond → __end__
  verify --fail ∧ retries left--> generate   (with typed feedback)
  verify --not grounded ∧ retries exhausted--> offer_ticket
  verify --advice drift ∧ retries exhausted--> respond_sanitized → __end__
  offer_ticket → [INTERRUPT: await user consent]
      --yes--> create_ticket → __end__
      --no--> graceful_end → __end__
```

### Node responsibilities (implement exactly this, nothing more)

| Node | Does | LLM? |
|---|---|---|
| `pre_checks` | Max input length, empty-input rejection, optional OpenAI moderation endpoint | No |
| `classify` | Structured-output call → `query_type ∈ {docs, account, both, off_topic, unsafe, prohibited_advice}` + `products` list. Labels built dynamically from the registry. Validate products against registry; invented labels → retry once, else decline. | Cheap model |
| `decline` | Return fixed template string per category. `off_topic` interpolates the product list. | **No LLM — templated only** |
| `retrieve_docs` | Top-k vector search in pgvector, scoped to the routed product's `doc_collection` from the registry | No |
| `retrieve_db` | (a) LLM picks a pre-written SQL template + fills typed params (shown only the routed product's templates); (b) deterministic executor validates template name, type-checks params via Pydantic, injects session `customer_id`, executes with bound params on a read-only role | Cheap model for (a) only |
| `generate` | Draft answer from retrieved context ONLY. Must emit inline citations (`[D1]`, `[DB]`) and the `INSUFFICIENT_CONTEXT` sentinel when context doesn't cover the question (sentinel → `offer_ticket`). On retry, append verifier feedback. | Strong model |
| `verify` | 3 checks: (1) claim-level groundedness LLM judge; (2) deterministic regex extraction of numbers/dates from draft asserted to literally appear in DB results; (3) advice-drift detection | Cheap model + deterministic code |
| `respond` | Format verified draft, render citations as sources | No |
| `respond_sanitized` | Deterministically drop sentences containing `advice_snippets`, append standard facts-not-recommendations line. Reached only when draft is grounded but advice drift survived all retries. | **No LLM** |
| `offer_ticket` | Inform user no reliable answer found; ask consent to raise ticket; LangGraph interrupt (state checkpointed, graph pauses; FastAPI resumes on same `thread_id`) | No |
| `create_ticket` | Call support-system API with full state: query, retrieved context, draft, verification failure report | No |

### Routing after verify (implement verbatim)

```python
def route_after_verify(state) -> str:
    v = state["verification"]
    if v.grounded and not v.contains_advice:
        return "respond"
    if not v.grounded:                                  # knowledge problem
        if state["retry_count"] < MAX_RETRIES:
            return "generate"                           # + unsupported-claims feedback
        return "offer_ticket"
    if state["retry_count"] < MAX_RETRIES:              # style problem (advice drift)
        return "generate"                               # + strip-advice feedback
    return "respond_sanitized"                          # NEVER ticket for advice drift
```

Typed retry feedback:

```python
def build_retry_feedback(v: VerifyResult) -> str:
    if not v.grounded:
        return (f"The following claims were not supported by the context, "
                f"remove or correct them: {v.unsupported_claims}")
    return (f"Remove all recommendation/suitability language. Do not suggest "
            f"what the user should choose or do. Restate facts only. "
            f"Offending snippets: {v.advice_snippets}")
```

`retry_count` is a **single shared counter**, incremented on every re-entry to `generate` regardless of failure type. Do not split into per-type budgets in v1.

### State schema

```python
class GraphState(TypedDict):
    query: str
    intent: str                    # docs|account|both|off_topic|unsafe|prohibited_advice
    products: list[str]
    doc_context: list[Chunk]       # chunk text + source metadata
    db_context: list[QueryResult]  # template_id + typed rows
    draft: str
    verification: VerifyResult
    retry_count: int
    decline_category: str | None
    ticket_id: str | None
```

```python
class ClassifyResult(BaseModel):
    query_type: Literal["docs", "account", "both",
                        "off_topic", "unsafe", "prohibited_advice"]
    products: list[str]

class VerifyResult(BaseModel):
    grounded: bool
    unsupported_claims: list[str]
    contains_advice: bool
    advice_snippets: list[str]
```

---

## 3. Hard invariants — NEVER violate these

1. **`customer_id` never passes through the LLM.** It comes from the authenticated FastAPI session and is injected by the DB executor. It is not in `GraphState`, not in any prompt, not fillable as a template param. A query like "show transactions for customer 999" still executes with the session's own ID.
2. **The LLM never writes SQL.** It only selects a template name and fills typed parameters. SQL text is fixed in `sql_templates.yaml`; execution uses bound parameters on a **read-only** DB role. `:customer_id` appears in every template's SQL but is never listed in `params`.
3. **No path from `generate` to the user bypasses `verify`.** Enforced by graph topology, not by prompt instructions.
4. **Answers come only from retrieved context** (doc chunks + DB results). No parametric LLM knowledge, no external information. Generation prompt enforces citations + `INSUFFICIENT_CONTEXT` sentinel.
5. **Decline responses are fixed templates, never LLM-generated.** Compliance-approvable, injection-proof.
6. **Advice drift never routes to `offer_ticket`.** It is a style failure; retries go back to `generate`, exhaustion goes to `respond_sanitized`. `offer_ticket` is reachable ONLY from groundedness exhaustion or `INSUFFICIENT_CONTEXT` — every ticket must represent a genuine knowledge gap.
7. **`off_topic` / `unsafe` / `prohibited_advice` never offer a ticket.** They go straight to templated decline → END.
8. **All product-specific facts live in configuration** (`products.yaml`, `sql_templates.yaml`), loaded once at startup by the registry. No product names hardcoded in nodes, prompts built dynamically from registry. `registry.db_tables` doubles as the table allowlist.
9. **Model tiering:** cheap/fast model for `classify` and `verify` (and template selection in `retrieve_db`); strong model for `generate` only. Model names configured in one place (`services/llm.py`).
10. **`respond_sanitized` is deterministic** — sentence removal, not an LLM rewrite (a rewrite could introduce new hallucinations).

---

## 4. Repository layout (build to this structure)

```
app/
  config/
    products.yaml            # product registry (per-client)
    sql_templates.yaml       # parameterized SQL library (per-client)
    decline_templates.py     # fixed, compliance-approved decline strings
    settings.py              # env, model names, MAX_RETRIES, top-k, etc.
  core/
    registry.py              # loads/validates products.yaml → ProductConfig objects
    models.py                # shared Pydantic models (ClassifyResult, VerifyResult, ...)
  retrieval/
    ingestion.py             # doc chunking + embedding → pgvector (offline job);
                             #   stamps chunk-metadata contract: product, source,
                             #   doc_version (from the doc's 'Last updated' header),
                             #   ingested_at, content_hash
    retriever.py             # collection-scoped vector search (metadata passes through)
  db/
    schema.sql               # 9 tables: customers + {fd,gold,bond,mf} holdings & transactions;
                             #   chatbot_readonly role (SELECT-only, 5s statement timeout)
    seed_dev.sql             # dev data incl. CUST-0001 (notebook test customer) and
                             #   CUST-0002 (data-isolation counterpart)
    templates.py             # loads/validates sql_templates.yaml
    executor.py              # type-check params, inject customer_id, bound-param
                             #   execution, read-only role, row limits
  verification/
    judge.py                 # LLM groundedness judge (structured output)
    numeric_check.py         # deterministic number/date matching vs DB results
    advice_check.py          # advice-drift detection (part of judge prompt + parse)
  graph/
    state.py                 # GraphState TypedDict
    nodes/
      pre_checks.py
      classify.py
      retrieve_docs.py
      retrieve_db.py
      generate.py
      verify.py
      respond.py             # respond + respond_sanitized
      decline.py
      tickets.py             # offer_ticket (interrupt) + create_ticket
    routing.py               # route_after_classify, route_after_verify
    builder.py               # graph assembly + PostgresSaver checkpointer + compile()
  services/
    tickets.py               # support-system API client
    llm.py                   # model clients: cheap (classify/verify) vs strong (generate)
  api/
    auth.py                  # session → customer_id (FastAPI dependency)
    routes.py                # POST /chat (invoke), POST /resume (interrupt resume)
    schemas.py               # request/response models
  observability/
    logging.py               # per-node structured logs, decline categories,
                             #   verify verdicts (→ future eval set)
ui/
  streamlit_app.py           # demo UI: chat + ticket-consent flow
notebooks/
  v1_module_tests.ipynb      # stage-by-stage interactive tests; one section per
                             #   tracker stage — run a module's cells before marking
                             #   it Done in docs/module_tracker.md
  ingestion_experiments.ipynb# module 1.4 experimentation: LangChain loaders +
                             #   RecursiveCharacterTextSplitter + OpenAIEmbeddings +
                             #   PGVector; settle chunking/embedding choices here,
                             #   then port into app/retrieval/ingestion.py
data/
  docs/<product_id>/         # raw product documentation, one folder per registry ID
tests/
  unit/                      # per-node tests, executor security tests, numeric_check
  eval/
    classify_set.jsonl       # 30–50 labeled queries across all 6 categories
    grounding_set.jsonl      # (context, draft, verdict) triples from logs
```

Registry pattern (reference implementation):

```python
class ProductConfig(BaseModel):
    display_name: str
    doc_collection: str
    db_tables: list[str]
    sql_templates: list[str]

class Registry:
    def __init__(self, path="config/products.yaml"):
        raw = yaml.safe_load(open(path))
        self.products = {k: ProductConfig(**v) for k, v in raw.items()}
    def get(self, product_id) -> ProductConfig: ...
    def product_ids(self) -> list[str]: ...
```

SQL template shape (reference):

```yaml
fd_transactions_by_date:
  description: "A customer's FD transactions within a date range"
  sql: |
    SELECT txn_id, fd_id, txn_type, amount, txn_date
    FROM fd_transactions
    WHERE customer_id = :customer_id
      AND txn_date BETWEEN :start_date AND :end_date
    ORDER BY txn_date DESC
  params:
    - {name: start_date, type: date}
    - {name: end_date, type: date}
```

---

## 5. Build order (6 stages, each independently demo-able)

1. **Registry + doc ingestion + plain RAG chain** — prove retrieval quality before any graph work. Ingestion (`ingestion.py`) is a **separate offline job**, not part of the request graph: experiment in `notebooks/ingestion_experiments.ipynb` (loaders, chunk size/overlap, embedding model), freeze choices into `settings.py`, then port into `app/retrieval/ingestion.py`. Re-run whenever product docs change.
2. **Graph skeleton:** classify → retrieve_docs → generate → verify → respond.
3. **DB templates + executor** (with security tests) — account queries.
4. **Interrupt-based ticket flow** + PostgresSaver checkpointing.
5. **Decline paths + advice-drift check + respond_sanitized.**
6. **Streamlit UI + LangSmith tracing.**

**Per-module workflow:** implement the module in `app/` → run its cells in
`notebooks/v1_module_tests.ipynb` (the tracker's "Notebook check" column maps each module
to its cells) → cells green → mark the row Done in `docs/module_tracker.md` → next module.
A stage is complete only when its notebook section passes end-to-end and its demo criterion
is met. Notebook cells import from `app/`, so they fail with ImportError until the module
exists — red → implement → green → next.

Track progress in `docs/module_tracker.md`.

---

## 6. Conventions & principles

- Every graph node is a small function on typed state → independently unit-testable. Keep nodes single-purpose.
- `graph/builder.py` should stay ~50 lines — it IS the architecture diagram in code.
- `verification/` and `db/executor.py` are the safety-critical modules → deepest test coverage. Executor security tests must cover: unknown template name rejection, param type violations, `customer_id` override attempts, SQL-injection-shaped param values.
- Structured outputs everywhere an LLM makes a decision (Pydantic models via `with_structured_output`). Validate; never trust unvalidated LLM output downstream.
- Log every decline with its category (`prohibited_advice` declines are product telemetry). Log every verify verdict as (context, draft, verdict) — this accumulates into `tests/eval/grounding_set.jsonl`.
- Build the classify eval set (~30–50 labeled queries, all 6 categories) **before** tuning the classify prompt. Hardest boundary to test: advice-adjacent factual questions ("which FD tenure has the highest rate?" = factual/docs; "which tenure should I pick?" = prohibited_advice).
- Ticket payload must include: query, retrieved context, draft, and why verification failed.
- **Chunk-metadata contract (ingestion):** every chunk carries `product`, `source`, `doc_version`, `ingested_at`, `content_hash`. `respond` renders "Source: <source>, updated <doc_version>" with citations; observability logs these fields with each verify verdict so any answer can be traced to the exact document version that supported it. KB docs must carry a `Last updated: <Month Year>` line in the document body — any format (.docx, .pdf, .md, .txt); `doc_version` is parsed from the loaded text, so the pipeline is format-agnostic.

## 7. Out of scope for v1 — do not build

- Text-to-SQL (templates only; migration trigger is documented in the reference doc §7)
- Agentic/ReAct retrieval loops, supervisor agents, computation tools
- Multi-query expansion / rerankers / query rewriting (documented as later in-graph fixes)
- Separate guardrail node before classify (Option B folds it into classify)
- Per-failure-type retry budgets (single shared `retry_count`)
- New infrastructure beyond PostgreSQL + pgvector
