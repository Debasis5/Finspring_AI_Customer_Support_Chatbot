"""Collection-scoped vector search over the pgvector store, with a config-drift guard.

The read side of the store that ``ingestion.py`` builds. ``retrieve_docs`` (module 2.5)
calls ``retrieve()`` with the product(s) ``classify`` routed to; chunk metadata passes
through untouched so ``respond`` can render "Source: <source>, updated <doc_version>" and
observability can log which document version supported an answer.

Product scoping is **structural, not filtered**: each product owns a collection
(``registry.get(pid).doc_collection``), so another product's chunks are never candidates
rather than being excluded by a ``WHERE`` clause a caller could forget. Callers pass a
``product_id``; no collection name appears outside the registry (invariant I-8).

Decisions settled in ``notebooks/experiments/1_5_retriever.ipynb`` against the real store:

**Top-k only — no distance threshold.** Over a 9-probe set with known answers, correct
chunks scored 0.332-0.709 and the best distractor per query 0.335-0.666; in three of nine
probes the correct chunk scored *worse* than the best distractor. Absolute cosine distance
tracks question-passage phrasing similarity, not correctness, so no cutoff separates them —
every candidate traded roughly one correct answer for one distractor. Filtering here would
also make retrieval failures look like sparse documentation, and the failure it would guard
against (junk reaching the customer) is already covered by ``generate``'s
``INSUFFICIENT_CONTEXT`` sentinel and the ``verify`` groundedness gate. Scores are returned
so relevance stays observable in logs and eval sets, never filtered on.

**``TOP_K`` stays 4.** Eight of nine probes retrieve their answer by rank 3. Raising k to 5
would rescue only the documented mutual-funds NAV gap — a KB content defect, fixed by
editing the doc and re-ingesting — while adding a fifth noisy chunk to every other query.

**The drift guard reads metadata, it does not embed.** See ``assert_store_matches``.

Nothing here defines a pipeline constant: ``TOP_K``, ``EMBEDDING_MODEL``, ``CHUNK_SIZE``
and ``CHUNK_OVERLAP`` all come from ``settings.py``, which is what makes the drift guard
meaningful — a constant redeclared here could differ from settings while the stamped
metadata still agreed with it.
"""

from __future__ import annotations

import logging
import warnings
from functools import cache

import sqlalchemy as sa
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector

from app.config.settings import settings
from app.core.registry import get_registry

log = logging.getLogger(__name__)


class StoreConfigMismatch(RuntimeError):
    """The store was built with different settings than the ones now configured.

    Fatal by design: continuing would query vectors that are not comparable to the query
    vector, and nothing downstream can detect that.
    """


# -- store access ----------------------------------------------------------


@cache
def _get_engine() -> sa.Engine:
    """SQLAlchemy engine for the drift guard's metadata reads.

    Separate from the PGVector connection because the guard deliberately runs *before* any
    vector work; see ``assert_store_matches``.
    """
    return sa.create_engine(settings.DATABASE_URL)


@cache
def _open_store(collection: str, embedding_model: str) -> PGVector:
    """Open one collection. Cached — constructing a PGVector opens a connection pool.

    ``embedding_model`` is a cache-key parameter so a settings change yields a new store
    rather than silently reusing one bound to the old embedder.
    """
    return PGVector(
        embeddings=OpenAIEmbeddings(model=embedding_model),
        collection_name=collection,
        connection=settings.DATABASE_URL,
    )


# -- drift guard (invariant I-11) ------------------------------------------

# Reads langchain-postgres' own tables. This couples us to its schema, which is a real
# cost, accepted deliberately: ingestion already depends on the same library owning these
# tables, and a layout change fails loudly here rather than silently degrading retrieval.
_BUILD_CONFIG_SQL = sa.text("""
    SELECT e.cmetadata
    FROM langchain_pg_embedding e
    JOIN langchain_pg_collection c ON e.collection_id = c.uuid
    WHERE c.name = :collection
    LIMIT 1
""")


def read_build_config(collection: str) -> dict | None:
    """One chunk's stamped metadata, or None if the collection is absent or empty.

    A plain column read: no query is embedded, so this cannot fail on vector dimensions
    and costs no OpenAI call.
    """
    with _get_engine().connect() as conn:
        row = conn.execute(_BUILD_CONFIG_SQL, {"collection": collection}).fetchone()
    return row[0] if row else None


@cache
def assert_store_matches(
    product_id: str, embedding_model: str, chunk_size: int, chunk_overlap: int
) -> None:
    """Verify one collection was built with the settings now configured.

    The sole enforcement point for invariant I-11.

    ``embedding_model`` mismatch **raises**: the query is embedded at query time, so
    vectors from a different model are not comparable. When dimensions differ, pgvector
    itself would reject the comparison — but when they happen to match (two 1536-dim
    models) it returns arbitrary chunks with **no error**, and every downstream gate still
    passes: ``verify`` asks whether the draft is grounded in the context it was given, and
    it is — in the garbage it was given. Silent corruption nothing else catches.

    ``chunk_size`` / ``chunk_overlap`` mismatch **warns**: nothing in the retrieval path
    reads them, so a mismatch means the store predates a settings change — chunks are
    stale, not wrong. Checked anyway because the realistic failure is a half-finished edit
    (settings tuned, ingestion not re-run) and the metadata is already fetched.

    Probes by reading metadata rather than by running a similarity search. An
    embedding-based probe must embed the *candidate* model to look anything up, which makes
    pgvector raise a raw ``DataError`` on a dimension mismatch before this function can
    compare anything — so callers catching ``StoreConfigMismatch`` would miss it, and the
    operator would get a vector-dimension stack trace instead of "re-run ingestion". Reading
    ``cmetadata`` directly makes both mismatch shapes raise the same exception. (The
    embedding-based prototype and this failure are recorded in the notebook, §5b.)

    Cached on all four arguments rather than reading ``settings`` internally, so one read
    runs per collection per process and a settings change re-probes instead of returning a
    stale pass. Both fixes are the same: re-run the offline ingestion job.
    """
    collection = get_registry().get(product_id).doc_collection
    meta = read_build_config(collection)
    if meta is None:
        raise StoreConfigMismatch(
            f"collection '{collection}' is empty or does not exist — run: "
            f"python -m app.retrieval.ingestion --product {product_id}"
        )

    built_with = meta.get("embedding_model")
    if built_with != embedding_model:
        raise StoreConfigMismatch(
            f"collection '{collection}' was built with embedding model '{built_with}', but "
            f"settings specify '{embedding_model}'. Vectors from different models are not "
            f"comparable — re-run: python -m app.retrieval.ingestion"
        )

    for field, current in (("chunk_size", chunk_size), ("chunk_overlap", chunk_overlap)):
        if meta.get(field) != current:
            warnings.warn(
                f"collection '{collection}' was built with {field}={meta.get(field)}, but "
                f"settings specify {current}. Chunks are stale (not wrong) — re-run "
                f"ingestion to sync.",
                stacklevel=2,
            )


# -- retrieval -------------------------------------------------------------


def retrieve_with_scores(
    query: str, product_id: str, k: int | None = None
) -> list[tuple[Document, float]]:
    """Top-k chunks for one product, with cosine distances (lower is closer).

    Scores are for observability — logging, eval sets, tuning — and are deliberately not
    filtered on; see the module docstring.

    Raises ``KeyError`` on an unknown product (via the registry, before any embedding
    spend) so an LLM-invented product label can never resolve to a default collection, and
    ``StoreConfigMismatch`` if the store drifted from settings.
    """
    assert_store_matches(
        product_id, settings.EMBEDDING_MODEL, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP
    )
    collection = get_registry().get(product_id).doc_collection
    store = _open_store(collection, settings.EMBEDDING_MODEL)
    results = store.similarity_search_with_score(query, k=k or settings.TOP_K)

    if results:
        log.debug(
            "retrieved %d chunk(s) from '%s' for %r (scores: %s)",
            len(results),
            collection,
            query[:60],
            ", ".join(f"{score:.3f}" for _, score in results),
        )
    else:
        log.warning("collection '%s' returned no chunks for %r", collection, query[:60])
    return results


def retrieve(query: str, product_id: str, k: int | None = None) -> list[Document]:
    """Top-k chunks for one product. The common case — scores dropped."""
    return [doc for doc, _ in retrieve_with_scores(query, product_id, k)]


def retrieve_many(query: str, product_ids: list[str], k: int | None = None) -> list[Document]:
    """Top-k chunks per product across several products, best-first overall.

    Used when ``classify`` routes one query to more than one product. Retrieves k from each
    collection and merges by score, so one product cannot crowd another out of the result.
    Cross-collection distances are not a reliable quality scale (module docstring), but
    merging on score is still a better ordering than concatenation, and every chunk carries
    ``product`` metadata regardless.
    """
    scored: list[tuple[Document, float]] = []
    for pid in product_ids:
        scored.extend(retrieve_with_scores(query, pid, k))
    scored.sort(key=lambda pair: pair[1])
    return [doc for doc, _ in scored]
