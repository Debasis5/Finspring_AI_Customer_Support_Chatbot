# v1 Module Tracker

Track implementation of Architecture 2 (v1). Update **Status** and **Notes** as you go.
Statuses: `⬜ Not started` · `🟨 In progress` · `✅ Done` · `🟥 Blocked`

**Workflow:** implement the module → run its cells in `notebooks/v1_module_tests.ipynb`
(the **Notebook check** column names the cell(s); the notebook's cells carry the same
module numbers in their header comments) → green → mark ✅ here → next module.
Rows marked *pytest* are covered by the unit test suite instead of the notebook.

Scope rule: v1 only. Anything in the reference doc's §7 (migration path) is out of scope here.

---

## Stage 1 — Registry + doc ingestion + plain RAG chain
*Goal: prove retrieval quality before any graph work.*

| # | Module / File | Deliverable | Notebook check | Status | Owner | Notes |
|---|---|---|---|---|---|---|
| 1.1 | `app/config/products.yaml` | Registry config for all 4 products (display_name, doc_collection, db_tables, sql_templates) | Stage 1 · "Registry loads and validates" | ⬜ | | |
| 1.2 | `app/core/registry.py` | Loads/validates products.yaml → ProductConfig objects; `get()`, `product_ids()` | Stage 1 · "Registry loads and validates" | ⬜ | | |
| 1.3 | `app/config/settings.py` | Env, model names, MAX_RETRIES, top-k | Stage 0 setup cells (env asserts) | ⬜ | | |
| 1.4 | `app/retrieval/ingestion.py` | **Separate offline job**: LangChain loaders → RecursiveCharacterTextSplitter → OpenAIEmbeddings → PGVector, one collection per product; chunk-metadata contract on every chunk: `product`, `source`, `doc_version` (from 'Last updated' header), `ingested_at`, `content_hash` | `ingestion_experiments.ipynb` (full run: load → chunk → ingest → metadata-contract + scoping + idempotency checks), then `v1_module_tests.ipynb` Stage 1 "Run ingestion" | ⬜ | | |
| 1.5 | `app/retrieval/retriever.py` | Collection-scoped top-k vector search | Stage 1 · retrieval smoke + scoping check (1.5b) | ⬜ | | |
| 1.6 | Plain RAG chain (throwaway/dev script) | Retrieval-quality smoke test over real product docs | Stage 1 · plain RAG + sentinel (1.6b) | ⬜ | | |
| 1.7 | `tests/unit/` — registry + retriever tests | Config validation, collection scoping | pytest (`tests/unit`) | ⬜ | | |

**Stage 1 demo:** ask doc questions per product, inspect retrieved chunks.

---

## Stage 2 — Graph skeleton (docs path)
*classify → retrieve_docs → generate → verify → respond*

| # | Module / File | Deliverable | Notebook check | Status | Owner | Notes |
|---|---|---|---|---|---|---|
| 2.1 | `app/core/models.py` | ClassifyResult, VerifyResult, Chunk, QueryResult | indirect — used by all Stage 2 cells | ⬜ | | |
| 2.2 | `app/graph/state.py` | GraphState TypedDict (per CLAUDE.md schema) | indirect — used by all Stage 2 cells | ⬜ | | |
| 2.3 | `app/services/llm.py` | Model tiering: cheap (classify/verify) vs strong (generate) | indirect — used by all Stage 2 cells | ⬜ | | |
| 2.4 | `app/graph/nodes/classify.py` | Structured output, 6 labels, registry-driven prompt, product validation (retry once → decline) | Stage 2 · boundary queries + product validation (2.4b) | ⬜ | | |
| 2.5 | `app/graph/nodes/retrieve_docs.py` | Registry-scoped retrieval into `doc_context` | Stage 2 · end-to-end docs run | ⬜ | | |
| 2.6 | `app/graph/nodes/generate.py` | Context-only prompt, `[D1]`/`[DB]` citations, `INSUFFICIENT_CONTEXT` sentinel, retry-feedback append | Stage 2 · end-to-end docs run + Stage 1 sentinel | ⬜ | | |
| 2.7 | `app/verification/judge.py` | Claim-level groundedness LLM judge (structured output) | Stage 2 · judge isolation (grounded vs fabricated) | ⬜ | | |
| 2.8 | `app/graph/nodes/verify.py` | Wires judge into graph, populates VerifyResult | Stage 2 · judge isolation + end-to-end docs run | ⬜ | | |
| 2.9 | `app/graph/nodes/respond.py` (respond only) | Format verified draft, render citations | Stage 2 · end-to-end docs run | ⬜ | | |
| 2.10 | `app/graph/routing.py` | route_after_classify, route_after_verify (verbatim per CLAUDE.md) | Stage 2 e2e + Stage 5 decline/drift cells | ⬜ | | |
| 2.11 | `app/graph/builder.py` | Graph assembly + compile (in-memory checkpointer OK at this stage) | Stage 2 · `draw_ascii()` topology vs reference §2 | ⬜ | | |
| 2.12 | `tests/eval/classify_set.jsonl` | 30–50 labeled queries across all 6 categories — **before tuning classify prompt** | Stage 2 · classify eval-set run | ⬜ | | |
| 2.13 | `tests/unit/` — classify, generate, verify, routing tests | Per-node tests on typed state | pytest (`tests/unit`) | ⬜ | | |

**Stage 2 demo:** end-to-end docs Q&A with verification gate + retry loop (`INSUFFICIENT_CONTEXT` can dead-end to a stub until Stage 4).

---

## Stage 3 — DB templates + executor (account path)

| # | Module / File | Deliverable | Notebook check | Status | Owner | Notes |
|---|---|---|---|---|---|---|
| 3.0 | `db/schema.sql` + `db/seed_dev.sql` | 9-table schema (customers + holdings/transactions per product), customer_id indexes, `chatbot_readonly` role with SELECT-only grants + statement timeout; dev seed incl. CUST-0001/CUST-0002 isolation pair | prerequisite — apply before 3.3 executor cells | ⬜ | | |
| 3.1 | `app/config/sql_templates.yaml` | Template library for all 4 products (holdings, transactions-by-date, maturity, etc.) — `:customer_id` in SQL, never in params | Stage 3 · template validation (`:customer_id` rule) | ⬜ | | |
| 3.2 | `app/db/templates.py` | Loads/validates sql_templates.yaml | Stage 3 · template validation | ⬜ | | |
| 3.3 | `app/db/executor.py` | Template-name validation, Pydantic param type-checks, session customer_id injection, bound params, read-only role, row limits | Stage 3 · executor happy path + SECURITY cells | ⬜ | | |
| 3.4 | `app/graph/nodes/retrieve_db.py` | LLM template selection (routed product's templates only) → executor → `db_context` | Stage 3 · select_and_execute + data-isolation cell | ⬜ | | |
| 3.5 | `app/verification/numeric_check.py` | Regex-extract numbers/dates from draft; assert literal presence in DB results | Stage 3 · numeric check (transposed digit) | ⬜ | | |
| 3.6 | Parallel branch wiring in `builder.py` | `both` → retrieve_docs ∥ retrieve_db | Stage 3 · e2e account + "both" parallel run | ⬜ | | |
| 3.7 | `tests/unit/` — **executor security tests** | Unknown template rejection; param type violations; customer_id override attempts; injection-shaped param values (e.g. `'; DROP TABLE--` as a date) | Stage 3 · SECURITY cells **and** pytest | ⬜ | | |
| 3.8 | `tests/unit/` — numeric_check tests | Transposed digits, dates, misses | Stage 3 · numeric check cell **and** pytest | ⬜ | | |

**Stage 3 demo:** account queries ("show my FD holdings", "gold transactions in January") answered from DB with numeric verification.

---

## Stage 4 — HITL ticket flow + checkpointing

| # | Module / File | Deliverable | Notebook check | Status | Owner | Notes |
|---|---|---|---|---|---|---|
| 4.1 | PostgresSaver checkpointer in `builder.py` | Replaces in-memory; state survives restarts / replicas | Stage 4 · checkpointer setup + restart-survival cell | ⬜ | | |
| 4.2 | `app/graph/nodes/tickets.py` — offer_ticket | Inform + consent ask + LangGraph interrupt | Stage 4 · interrupt trigger + `get_state()` | ⬜ | | |
| 4.3 | `app/services/tickets.py` | Support-system API client | Stage 4 · resume consent=yes (ticket created) | ⬜ | | |
| 4.4 | `app/graph/nodes/tickets.py` — create_ticket | Rich payload: query, retrieved context, draft, verification failure report | Stage 4 · resume consent=yes — inspect payload | ⬜ | | |
| 4.5 | `app/api/auth.py` | Session → customer_id FastAPI dependency | manual API test (notebook resumes graph directly) | ⬜ | | |
| 4.6 | `app/api/routes.py` + `schemas.py` | POST /chat (invoke), POST /resume (interrupt resume on thread_id) | Stage 4 resume cells + manual API test (/chat, /resume) | ⬜ | | |
| 4.7 | `tests/unit/` — interrupt/resume tests | Consent yes → create_ticket; no → graceful_end; resume on same thread_id | pytest (`tests/unit`) | ⬜ | | |

**Stage 4 demo:** unanswerable in-scope question → ticket offer → pause → consent → ticket created with full context.

---

## Stage 5 — Guardrails complete

| # | Module / File | Deliverable | Notebook check | Status | Owner | Notes |
|---|---|---|---|---|---|---|
| 5.1 | `app/graph/nodes/pre_checks.py` | Max length, empty input, optional moderation endpoint | Stage 5 · pre_checks cell | ⬜ | | |
| 5.2 | `app/config/decline_templates.py` | Fixed strings for off_topic / unsafe / prohibited_advice (off_topic interpolates product list) | Stage 5 · decline cases (all 3 categories) | ⬜ | | |
| 5.3 | `app/graph/nodes/decline.py` | Templated decline node (no LLM), logs decline_category | Stage 5 · decline cases (asserts no ticket) | ⬜ | | |
| 5.4 | `app/verification/advice_check.py` | Advice-drift detection → contains_advice + advice_snippets | Stage 5 · advice-drift detection isolation | ⬜ | | |
| 5.5 | `app/graph/nodes/respond.py` — respond_sanitized | Deterministic sentence removal of advice_snippets + standard facts line | Stage 5 · sanitize_draft cell (facts preserved) | ⬜ | | |
| 5.6 | Retry-feedback split (`build_retry_feedback`) | Groundedness feedback vs strip-advice feedback | Stage 5 · drift retry loop through graph | ⬜ | | |
| 5.7 | `tests/unit/` — decline routing + respond_sanitized tests | All 3 decline categories → END (never ticket); advice-drift exhaustion → sanitized, never ticket | pytest (`tests/unit`) | ⬜ | | |
| 5.8 | Classify boundary evals passing | Advice-adjacent factual vs prohibited_advice cases from classify_set.jsonl | Stage 5 · classify eval-set regression re-run | ⬜ | | |

**Stage 5 demo:** "what's the weather" / "help me hack" / "should I buy gold or FD?" → correct templated declines; advice drift in a factual answer → auto-corrected on retry.

---

## Stage 6 — UI + observability

| # | Module / File | Deliverable | Notebook check | Status | Owner | Notes |
|---|---|---|---|---|---|---|
| 6.1 | `ui/streamlit_app.py` | Chat UI incl. ticket-consent flow (resume path) | manual — run Streamlit (incl. consent flow) | ⬜ | | |
| 6.2 | `app/observability/logging.py` | Per-node structured logs; decline categories; verify verdicts | Stage 6 · grounding-triples accumulation cell | ⬜ | | |
| 6.3 | LangSmith tracing | Node-transition traces enabled | Stage 6 · LangSmith tracing sanity cell | ⬜ | | |
| 6.4 | `tests/eval/grounding_set.jsonl` pipeline | Verify verdicts (context, draft, verdict) accumulating from logs | Stage 6 · grounding-triples accumulation cell | ⬜ | | |

**Stage 6 demo:** full end-to-end demo through Streamlit with traceable runs.

---

## Cross-cutting invariant checks (verify before calling v1 done)

| # | Invariant | Verified by | Status |
|---|---|---|---|
| I-1 | customer_id never in LLM prompts/state; injected by executor only | Code review + executor tests (3.7) + notebook Final-gate I-1 cell | ⬜ |
| I-2 | LLM never writes SQL; bound params; read-only role | Executor tests (3.7) + notebook Final-gate I-2 cells | ⬜ |
| I-3 | No generate→user path bypasses verify | Graph topology review (builder.py) | ⬜ |
| I-4 | Answers from retrieved context only; sentinel honored | Generate tests (2.13) + grounding evals | ⬜ |
| I-5 | Declines are fixed templates, no LLM | decline.py review + tests (5.7) | ⬜ |
| I-6 | Advice drift never reaches offer_ticket; tickets only from groundedness exhaustion / INSUFFICIENT_CONTEXT | Routing tests (2.13, 5.7) + notebook Stage 5 drift cell | ⬜ |
| I-7 | off_topic/unsafe/prohibited_advice never offer tickets | Routing tests (5.7) + notebook Final-gate I-7 cell | ⬜ |
| I-8 | No hardcoded product names in nodes/prompts; all via registry | Code review | ⬜ |
| I-9 | Model tiering configured only in services/llm.py | Code review | ⬜ |
| I-10 | respond_sanitized is deterministic (no LLM rewrite) | Code review + tests (5.7) | ⬜ |