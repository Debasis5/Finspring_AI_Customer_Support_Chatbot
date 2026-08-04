# v1 Module Tracker

Track implementation of Architecture 2 (v1). Update **Status** and **Notes** as you go.
Statuses: `⬜ Not started` · `🟨 In progress` · `✅ Done` · `🟥 Blocked`

**Workflow — notebook first** (CLAUDE.md §5): prototype the module in
`notebooks/experiments/<stage>_<module>_<slug>.ipynb` against real data → port the settled
version into `app/` → add clean check cells to `notebooks/v1_module_tests.ipynb` (the
**Notebook check** column names them; cells carry the same module numbers in their header
comments) → green → mark ✅ here → next module. Rows marked *pytest* are covered by the
unit test suite instead of the notebook.

The **Experiment nb** column tracks step 1. `—` means no experiment notebook is expected
(pure-config or test-only rows). Experiment notebooks are scratch: they may go stale after
porting and are not part of the regression suite.

Scope rule: v1 only. Anything in the reference doc's §7 (migration path) is out of scope here.

---

## Stage 1 — Registry + doc ingestion + plain RAG chain
*Goal: prove retrieval quality before any graph work.*

| # | Module / File | Deliverable | Experiment nb | Notebook check | Status | Notes |
|---|---|---|---|---|---|---|
| 1.1 | `app/config/products.yaml` | Registry config for all 4 products (display_name, doc_collection, db_tables, sql_templates) | — (pure config) | Stage 1 · "Registry loads and validates" | ✅ | 4 products keyed to `data/docs/` folder names. Tables follow `<prefix>_holdings`/`_transactions` (CLAUDE.md §4's 9-table count), not the reference doc's illustrative `fd_accounts`. `customers` deliberately excluded — it's cross-product, so it belongs in the executor's global allowlist, not one product's. Template names are Stage-3 placeholders; 3.1 must define exactly these. |
| 1.2 | `app/core/registry.py` | Loads/validates products.yaml → ProductConfig objects; `get()`, `product_ids()` | — (predates convention; built direct) | Stage 1 · "Registry loads and validates" | ✅ | Verified by direct run (notebook check pending 1.4). Adds `display_names()` (5.2 decline), `templates_for()` (3.4), `all_db_tables()` (3.3 allowlist), `is_valid_product()` (2.4). `extra="forbid"` so a typo'd key fails loudly; cross-product uniqueness enforced on `doc_collection`/`sql_templates`; `get()` raises on unknown IDs so invented labels can't resolve. 13 failure modes tested → all `RegistryError`. |
| 1.3 | `app/config/settings.py` | Env, model names, MAX_RETRIES, top-k, chunking + separators | `experiments/1_4_ingestion.ipynb` (values frozen there) | Stage 0 setup cells (env asserts) | ✅ | **Amended during 1.4:** added `SPLIT_SEPARATORS`/`SPLIT_IS_REGEX` (section-aware chunking — regex on numbered headings; measured across all 4 KB docs: multi-section chunks 9→1, chunks starting at a heading 6→41). `EMBEDDING_MODEL`/`CHUNK_SIZE`/`CHUNK_OVERLAP` unchanged at `text-embedding-3-small`/1000/150 — 1000/150 now **validated** by notebook comparison against 500/50 and 1500/200 (500/50 rejected: paragraphs average ~135 chars, so a 50-char overlap spans none and 17/25 boundaries carried nothing). — Verified by direct run — **`v1_module_tests.ipynb` does not exist yet**; re-confirm via its Stage 0 cells when created. Resolved `PGVECTOR_CONN` vs `DATABASE_URL` → `DATABASE_URL` (fixed cell 2 of `experiments/1_4_ingestion.ipynb`, which read a var defined nowhere). Models: `gpt-4o-mini`/`gpt-4o`, named here only (I-9). Frozen from notebook: chunk 1000/150, `text-embedding-3-small`. `OPENAI_API_KEY` needs `min_length=1` (empty string satisfied a bare `str`); singleton made lazy so the module imports without a populated `.env`. 7 validators tested. |
| 1.4 | `app/retrieval/ingestion.py` | **Separate offline job**, runnable as `python -m app.retrieval.ingestion`: LangChain loaders → RecursiveCharacterTextSplitter → OpenAIEmbeddings → PGVector, one collection per product. All parameters read from `settings.py` — no pipeline constants in the module. 8-field chunk-metadata contract: *provenance* `product`, `source`, `doc_version` (from 'Last updated' header), `ingested_at`, `content_hash` (load time) + *build config* `embedding_model`, `chunk_size`, `chunk_overlap` (chunk time). Skips Word lock files (`~$*.docx`) and logs every ignored file. **Raises if zero documents are found across all products** — a run that writes nothing is a broken job, not a no-op; a single empty product stays a warning (a client mid-onboarding may have only some products documented) | `experiments/1_4_ingestion.ipynb` ✅ (the worked example of the notebook-first workflow) | `v1_module_tests.ipynb` Stage 1 "Run ingestion" | ✅ | Verified by direct run — **`v1_module_tests.ipynb` does not exist yet**; re-confirm via its Stage 1 cells when created (planned at the end of Stage 1, covering 1.4–1.6 in one pass). `experiments/1_4_ingestion.ipynb` is fully green, drift-guard cell included. Live run writes 64 chunks (fd 16 / gold 16 / bonds 19 / mf 13), matching the notebook exactly. **`ingested_at` hoisted to `ingest_all`** — the notebook computed it inside `load_product_docs`, giving four products four timestamps seconds apart; the store now holds 1 distinct stamp per run, so a run is identifiable. Verified: 2nd full run leaves 64 chunks and one stamp (replace, not append); zero-docs and unknown-product both exit 1 (a cron needs a non-zero code — the tracker rule is worthless if the job exits 0); `~$*.docx` skipped and logged. Adds beyond spec: `--product` (repeatable), `--dry-run` (load+chunk, no embedding spend — catches doc/chunking errors for free), `_assert_contract` fail-before-write. `datetime.UTC` over `timezone.utc` (py313, ruff UP017). ruff clean. |
| 1.5 | `app/retrieval/retriever.py` | Collection-scoped top-k vector search + `assert_store_matches` startup drift guard (cached per collection): `embedding_model` mismatch raises, chunk params warn | `experiments/1_5_retriever.ipynb` ⬜ — **next up**. Open questions to settle there: (a) is `TOP_K=4` right? the mutual-funds NAV clause ranks below it (see gap note); (b) add a distance threshold, or top-k only? (c) caching shape for the per-collection probe. Cell 27 of `experiments/1_4_ingestion.ipynb` is a working drift-guard prototype to start from | Stage 1 · retrieval smoke + scoping check (1.5b) + drift-guard cell | ⬜ | |
| 1.6 | Plain RAG chain (throwaway/dev script) | Retrieval-quality smoke test over real product docs | `experiments/1_6_rag_chain.ipynb` ⬜ | Stage 1 · plain RAG + sentinel (1.6b) | ⬜ | |
| 1.7 | `tests/unit/` — registry + retriever tests | Config validation, collection scoping, drift guard (embedding-model mismatch raises; chunk-param mismatch warns but still returns) | — (test-only row) | pytest (`tests/unit`) | ⬜ | |

**Stage 1 demo:** ask doc questions per product, inspect retrieved chunks.

**KB content gap — `mutual_funds_kb.docx` has no NAV-calculation section** (⬜ open, found
during 1.4; fix is a doc edit, not code). "How is NAV calculated?" retrieves at 0.585–0.682
(cosine distance, lower is better) against 0.332–0.545 for `digital_fd`, and its top hit is
the glossary line "NAV: Net Asset Value per unit, computed daily" — the term, not the
method. The formula *does* exist — "computed daily as (portfolio value − liabilities) ÷
units outstanding" — but only as a subordinate clause inside chunk 2, headed
`1. What is a mutual fund?`, whose embedding is dominated by fund structure and SEBI
regulation. It therefore ranks below `TOP_K=4` and never reaches `generate`, which correctly
emits `INSUFFICIENT_CONTEXT` → `offer_ticket`. **The architecture behaves correctly; the
customer just doesn't get an answer the KB technically contains.** Not a chunking or
retrieval bug — chunk 2 is one coherent numbered section, and tuning `CHUNK_SIZE` to rescue
one buried sentence would degrade the other three products, which all retrieve on-topic
sections. Fix: add a numbered NAV section to the doc, then
`python -m app.retrieval.ingestion --product mutual_funds`; the section-aware splitter will
give it its own heading-anchored chunk. Worth capturing as an eval case — a question that
looks answerable from the KB but isn't is exactly the boundary the Stage 1 retrieval checks
and `tests/eval/` should cover.

**Environment (prerequisite for 1.4, not a tracker row):** dependencies pinned in
`pyproject.toml` + `uv.lock` (184 pkgs, `uv sync` reproduces); pytest configured with an
`integration` marker for tests needing live Postgres/OpenAI (`-m 'not integration'` to skip);
ruff configured (line-length 100, py313). pgvector container healthy on host port 5433.
`pypdf` was missing — `PyPDFLoader` imports without it but fails at load time; added, since
current KB docs are all `.docx` and it would only have surfaced on a client's first PDF.
`OPENAI_API_KEY` is set — 1.4 is unblocked.

---

## Stage 2 — Graph skeleton (docs path)
*classify → retrieve_docs → generate → verify → respond*

| # | Module / File | Deliverable | Experiment nb | Notebook check | Status | Notes |
|---|---|---|---|---|---|---|
| 2.1 | `app/core/models.py` | ClassifyResult, VerifyResult, Chunk, QueryResult | — (plain Pydantic defs; exercised by 2.4/2.7 notebooks) | indirect — used by all Stage 2 cells | ⬜ | | |. 
| 2.2 | `app/graph/state.py` | GraphState TypedDict (per CLAUDE.md schema) | — (config/test-only) | indirect — used by all Stage 2 cells | ⬜ | | |. 
| 2.3 | `app/services/llm.py` | Model tiering: cheap (classify/verify) vs strong (generate) | `experiments/2_3_llm_tiering.ipynb` ⬜ | indirect — used by all Stage 2 cells | ⬜ | | |. 
| 2.4 | `app/graph/nodes/classify.py` | Structured output, 6 labels, registry-driven prompt, product validation (retry once → decline) | `experiments/2_4_classify.ipynb` ⬜ | Stage 2 · boundary queries + product validation (2.4b) | ⬜ | | |. 
| 2.5 | `app/graph/nodes/retrieve_docs.py` | Registry-scoped retrieval into `doc_context` | `experiments/2_5_retrieve_docs.ipynb` ⬜ | Stage 2 · end-to-end docs run | ⬜ | | |. 
| 2.6 | `app/graph/nodes/generate.py` | Context-only prompt, `[D1]`/`[DB]` citations, `INSUFFICIENT_CONTEXT` sentinel, retry-feedback append | `experiments/2_6_generate.ipynb` ⬜ | Stage 2 · end-to-end docs run + Stage 1 sentinel | ⬜ | | |. 
| 2.7 | `app/verification/judge.py` | Claim-level groundedness LLM judge (structured output) | `experiments/2_7_judge.ipynb` ⬜ | Stage 2 · judge isolation (grounded vs fabricated) | ⬜ | | |. 
| 2.8 | `app/graph/nodes/verify.py` | Wires judge into graph, populates VerifyResult | `experiments/2_8_verify.ipynb` ⬜ | Stage 2 · judge isolation + end-to-end docs run | ⬜ | | |. 
| 2.9 | `app/graph/nodes/respond.py` (respond only) | Format verified draft, render citations | `experiments/2_9_respond.ipynb` ⬜ | Stage 2 · end-to-end docs run | ⬜ | | |. 
| 2.10 | `app/graph/routing.py` | route_after_classify, route_after_verify (verbatim per CLAUDE.md) | `experiments/2_10_routing.ipynb` ⬜ | Stage 2 e2e + Stage 5 decline/drift cells | ⬜ | | |. 
| 2.11 | `app/graph/builder.py` | Graph assembly + compile (in-memory checkpointer OK at this stage) | `experiments/2_11_builder.ipynb` ⬜ | Stage 2 · `draw_ascii()` topology vs reference §2 | ⬜ | | |. 
| 2.12 | `tests/eval/classify_set.jsonl` | 30–50 labeled queries across all 6 categories — **before tuning classify prompt** | — (config/test-only) | Stage 2 · classify eval-set run | ⬜ | | |. 
| 2.13 | `tests/unit/` — classify, generate, verify, routing tests | Per-node tests on typed state | — (config/test-only) | pytest (`tests/unit`) | ⬜ | | |. 

**Stage 2 demo:** end-to-end docs Q&A with verification gate + retry loop (`INSUFFICIENT_CONTEXT` can dead-end to a stub until Stage 4).

---

## Stage 3 — DB templates + executor (account path)

| # | Module / File | Deliverable | Experiment nb | Notebook check | Status | Notes |
|---|---|---|---|---|---|---|
| 3.0 | `db/schema.sql` + `db/seed_dev.sql` | 9-table schema (customers + holdings/transactions per product), customer_id indexes, `chatbot_readonly` role with SELECT-only grants + statement timeout; dev seed incl. CUST-0001/CUST-0002 isolation pair | — (config/test-only) | prerequisite — apply before 3.3 executor cells | ⬜ | | |. 
| 3.1 | `app/config/sql_templates.yaml` | Template library for all 4 products (holdings, transactions-by-date, maturity, etc.) — `:customer_id` in SQL, never in params | — (config/test-only) | Stage 3 · template validation (`:customer_id` rule) | ⬜ | | |. 
| 3.2 | `app/db/templates.py` | Loads/validates sql_templates.yaml | `experiments/3_2_templates.ipynb` ⬜ | Stage 3 · template validation | ⬜ | | |. 
| 3.3 | `app/db/executor.py` | Template-name validation, Pydantic param type-checks, session customer_id injection, bound params, read-only role, row limits | `experiments/3_3_executor.ipynb` ⬜ | Stage 3 · executor happy path + SECURITY cells | ⬜ | | |. 
| 3.4 | `app/graph/nodes/retrieve_db.py` | LLM template selection (routed product's templates only) → executor → `db_context` | `experiments/3_4_retrieve_db.ipynb` ⬜ | Stage 3 · select_and_execute + data-isolation cell | ⬜ | | |. 
| 3.5 | `app/verification/numeric_check.py` | Regex-extract numbers/dates from draft; assert literal presence in DB results | `experiments/3_5_numeric_check.ipynb` ⬜ | Stage 3 · numeric check (transposed digit) | ⬜ | | |. 
| 3.6 | Parallel branch wiring in `builder.py` | `both` → retrieve_docs ∥ retrieve_db | `experiments/3_6_parallel_branch.ipynb` ⬜ | Stage 3 · e2e account + "both" parallel run | ⬜ | | |. 
| 3.7 | `tests/unit/` — **executor security tests** | Unknown template rejection; param type violations; customer_id override attempts; injection-shaped param values (e.g. `'; DROP TABLE--` as a date) | — (config/test-only) | Stage 3 · SECURITY cells **and** pytest | ⬜ | | |. 
| 3.8 | `tests/unit/` — numeric_check tests | Transposed digits, dates, misses | — (config/test-only) | Stage 3 · numeric check cell **and** pytest | ⬜ | | |. 

**Stage 3 demo:** account queries ("show my FD holdings", "gold transactions in January") answered from DB with numeric verification.

---

## Stage 4 — HITL ticket flow + checkpointing

| # | Module / File | Deliverable | Experiment nb | Notebook check | Status | Notes |
|---|---|---|---|---|---|---|
| 4.1 | PostgresSaver checkpointer in `builder.py` | Replaces in-memory; state survives restarts / replicas | `experiments/4_1_checkpointer.ipynb` ⬜ | Stage 4 · checkpointer setup + restart-survival cell | ⬜ | | |. 
| 4.2 | `app/graph/nodes/tickets.py` — offer_ticket | Inform + consent ask + LangGraph interrupt | `experiments/4_2_offer_ticket.ipynb` ⬜ | Stage 4 · interrupt trigger + `get_state()` | ⬜ | | |. 
| 4.3 | `app/services/tickets.py` | Support-system API client | `experiments/4_3_ticket_client.ipynb` ⬜ | Stage 4 · resume consent=yes (ticket created) | ⬜ | | |. 
| 4.4 | `app/graph/nodes/tickets.py` — create_ticket | Rich payload: query, retrieved context, draft, verification failure report | `experiments/4_4_create_ticket.ipynb` ⬜ | Stage 4 · resume consent=yes — inspect payload | ⬜ | | |. 
| 4.5 | `app/api/auth.py` | Session → customer_id FastAPI dependency | `experiments/4_5_auth.ipynb` ⬜ | manual API test (notebook resumes graph directly) | ⬜ | | |. 
| 4.6 | `app/api/routes.py` + `schemas.py` | POST /chat (invoke), POST /resume (interrupt resume on thread_id) | `experiments/4_6_api_routes.ipynb` ⬜ | Stage 4 resume cells + manual API test (/chat, /resume) | ⬜ | | |. 
| 4.7 | `tests/unit/` — interrupt/resume tests | Consent yes → create_ticket; no → graceful_end; resume on same thread_id | — (config/test-only) | pytest (`tests/unit`) | ⬜ | | |. 

**Stage 4 demo:** unanswerable in-scope question → ticket offer → pause → consent → ticket created with full context.

---

## Stage 5 — Guardrails complete

| # | Module / File | Deliverable | Experiment nb | Notebook check | Status | Notes |
|---|---|---|---|---|---|---|
| 5.1 | `app/graph/nodes/pre_checks.py` | Max length, empty input, optional moderation endpoint | `experiments/5_1_pre_checks.ipynb` ⬜ | Stage 5 · pre_checks cell | ⬜ | | |. 
| 5.2 | `app/config/decline_templates.py` | Fixed strings for off_topic / unsafe / prohibited_advice (off_topic interpolates product list) | — (config/test-only) | Stage 5 · decline cases (all 3 categories) | ⬜ | | |. 
| 5.3 | `app/graph/nodes/decline.py` | Templated decline node (no LLM), logs decline_category | `experiments/5_3_decline.ipynb` ⬜ | Stage 5 · decline cases (asserts no ticket) | ⬜ | | |. 
| 5.4 | `app/verification/advice_check.py` | Advice-drift detection → contains_advice + advice_snippets | `experiments/5_4_advice_check.ipynb` ⬜ | Stage 5 · advice-drift detection isolation | ⬜ | | |. 
| 5.5 | `app/graph/nodes/respond.py` — respond_sanitized | Deterministic sentence removal of advice_snippets + standard facts line | `experiments/5_5_respond_sanitized.ipynb` ⬜ | Stage 5 · sanitize_draft cell (facts preserved) | ⬜ | | |. 
| 5.6 | Retry-feedback split (`build_retry_feedback`) | Groundedness feedback vs strip-advice feedback | `experiments/5_6_retry_feedback.ipynb` ⬜ | Stage 5 · drift retry loop through graph | ⬜ | | |. 
| 5.7 | `tests/unit/` — decline routing + respond_sanitized tests | All 3 decline categories → END (never ticket); advice-drift exhaustion → sanitized, never ticket | — (config/test-only) | pytest (`tests/unit`) | ⬜ | | |. 
| 5.8 | Classify boundary evals passing | Advice-adjacent factual vs prohibited_advice cases from classify_set.jsonl | — (config/test-only) | Stage 5 · classify eval-set regression re-run | ⬜ | | |. 

**Stage 5 demo:** "what's the weather" / "help me hack" / "should I buy gold or FD?" → correct templated declines; advice drift in a factual answer → auto-corrected on retry.

---

## Stage 6 — UI + observability

| # | Module / File | Deliverable | Experiment nb | Notebook check | Status | Notes |
|---|---|---|---|---|---|---|
| 6.1 | `ui/streamlit_app.py` | Chat UI incl. ticket-consent flow (resume path) | — (UI runs as a script, not a notebook; prototype by running Streamlit) | manual — run Streamlit (incl. consent flow) | ⬜ | | |. 
| 6.2 | `app/observability/logging.py` | Per-node structured logs; decline categories; verify verdicts | `experiments/6_2_logging.ipynb` ⬜ | Stage 6 · grounding-triples accumulation cell | ⬜ | | |. 
| 6.3 | LangSmith tracing | Node-transition traces enabled | — (config/test-only) | Stage 6 · LangSmith tracing sanity cell | ⬜ | | |. 
| 6.4 | `tests/eval/grounding_set.jsonl` pipeline | Verify verdicts (context, draft, verdict) accumulating from logs | `experiments/6_4_grounding_set.ipynb` ⬜ | Stage 6 · grounding-triples accumulation cell | ⬜ | | |. 
| 6.5 | Ingestion **run-level** reporting | One structured record per `ingest_all()` invocation: timestamp, duration, per-product file/chunk counts + collection, build config used, per-file `source`/`doc_version`/`content_hash`, skipped files, outcome. Deferred here deliberately (decided during 1.4) so it can match 6.2's formatters instead of inventing its own. **Not** per-node request logging — ingestion is a batch job with no `thread_id` or customer, so it does not belong in the request-path logger. Closes the `content_hash` audit loop: chunk metadata says which doc version supported an answer, this says when that version entered the store | `experiments/6_5_ingestion_reporting.ipynb` ⬜ | manual — run the offline job, inspect the emitted record | ⬜ | | |. 

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
| I-11 | Vector store can never be queried with an embedding model other than the one that built it | Drift guard in retriever.py (1.5) + tests (1.7) + notebook drift-guard cell | ⬜ |