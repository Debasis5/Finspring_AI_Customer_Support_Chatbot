"""Document ingestion — docs on disk → chunks → embeddings → pgvector.

**A separate offline job, not part of the request graph.** A human or a cron invokes it::

    python -m app.retrieval.ingestion              # all products
    python -m app.retrieval.ingestion --product bonds
    python -m app.retrieval.ingestion --dry-run    # load + chunk, no embedding calls

Re-run whenever product documentation changes **or** whenever ``EMBEDDING_MODEL``,
``CHUNK_SIZE``, or ``CHUNK_OVERLAP`` changes in ``settings.py`` — the store is built from
those settings, and changing them without re-ingesting leaves the store and the config
disagreeing (see ``retriever.assert_store_matches``).

Every pipeline parameter is read from ``settings.py``; this module defines none of its own.
That single source is what makes the drift guard meaningful — a constant redeclared here
could differ from settings while the stamped metadata still agreed with it.

Chunk-metadata contract — eight fields in two groups:

*Provenance* (stamped at load time, describes the source file)
    ``product``, ``source``, ``doc_version``, ``ingested_at``, ``content_hash``.
    Consumed by ``retrieve_docs`` → ``doc_context``, rendered by ``respond`` as
    "Source: <source>, updated <doc_version>", and logged with every verify verdict so an
    answer traces back to the exact document version that supported it.

*Build config* (stamped at chunk time, describes how the vector was produced)
    ``embedding_model``, ``chunk_size``, ``chunk_overlap``. Consumed by ``retriever.py``'s
    startup drift guard.

Decisions frozen in ``notebooks/ingestion_experiments.ipynb``; see that notebook for the
measurements behind the chunking parameters.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config.settings import settings
from app.core.registry import Registry, get_registry

log = logging.getLogger(__name__)

# One loader per supported extension. The keys double as the ingestable-suffix set, so
# adding a format here is the only change needed to support it.
LOADERS: dict[str, type] = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".md": TextLoader,
    ".txt": TextLoader,
}

PROVENANCE_META = frozenset({"product", "source", "doc_version", "ingested_at", "content_hash"})
BUILD_CONFIG_META = frozenset({"embedding_model", "chunk_size", "chunk_overlap"})
FULL_META = PROVENANCE_META | BUILD_CONFIG_META

# Word/Office lock files ("~$name.docx") match .docx but are not documents, and
# Docx2txtLoader fails on them. A KB file left open in Word would otherwise break the
# whole run with a confusing error.
LOCK_FILE_PREFIX = "~$"

_LAST_UPDATED_RE = re.compile(r"Last updated:\**\s*([A-Za-z]+\s+\d{4})")
UNKNOWN_VERSION = "unknown"


class IngestionError(RuntimeError):
    """Raised when the job cannot produce a usable store.

    Always fatal: ingestion is a batch job with no user waiting on it, so a partial or
    empty result must fail loudly rather than leave a half-built store behind.
    """


# -- discovery -------------------------------------------------------------


def is_ingestable(path: Path) -> bool:
    """True for real product documents, excluding Office lock files."""
    return (
        path.is_file()
        and path.suffix.lower() in LOADERS
        and not path.name.startswith(LOCK_FILE_PREFIX)
    )


def discover_documents(product_id: str, docs_root: Path | None = None) -> list[Path]:
    """Files to ingest for one product, from ``<docs_root>/<product_id>/``.

    Every skipped file is logged: a doc silently missing from the store is invisible until
    a customer asks about it and retrieval comes back empty.
    """
    root = docs_root or settings.DOCS_ROOT
    folder = root / product_id
    if not folder.exists():
        log.warning("%s: no docs folder at %s", product_id, folder)
        return []

    files, ignored = [], []
    for path in sorted(folder.iterdir()):
        if not path.is_file() or path.name == ".gitkeep":
            continue
        (files if is_ingestable(path) else ignored).append(path)

    for path in ignored:
        log.warning("%s: ignoring %s (not an ingestable document)", product_id, path.name)
    return files


# -- loading ---------------------------------------------------------------


def parse_doc_version(text: str) -> str:
    """Extract the doc's ``Last updated: <Month Year>`` line.

    Read from the *loaded* text rather than the raw bytes, so the pipeline is
    format-agnostic — the header only has to survive text extraction, whatever the format.
    """
    match = _LAST_UPDATED_RE.search(text)
    return match.group(1) if match else UNKNOWN_VERSION


def load_product_docs(
    product_id: str, ingested_at: str, docs_root: Path | None = None
) -> list[Document]:
    """Load one product's documents and stamp the provenance half of the contract.

    ``ingested_at`` is passed in rather than computed here so every chunk written by a
    single run shares one timestamp, making the run identifiable in the store.
    """
    docs: list[Document] = []
    for path in discover_documents(product_id, docs_root):
        content_hash = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
        loaded = LOADERS[path.suffix.lower()](str(path)).load()

        doc_version = parse_doc_version("\n".join(d.page_content for d in loaded))
        if doc_version == UNKNOWN_VERSION:
            # Not fatal, but it renders as "updated unknown" in every citation this
            # document ever supports. Fix the doc's header, or the regex.
            log.warning(
                "%s: no 'Last updated:' header found in %s — citations will read 'updated unknown'",
                product_id,
                path.name,
            )

        for doc in loaded:
            doc.metadata.update(
                product=product_id,
                source=path.name,
                doc_version=doc_version,
                ingested_at=ingested_at,
                content_hash=content_hash,
            )
        docs.extend(loaded)
        log.info(
            "%s: loaded %s (%d part(s), version %s)",
            product_id,
            path.name,
            len(loaded),
            doc_version,
        )
    return docs


# -- chunking + embedding --------------------------------------------------


def build_splitter() -> RecursiveCharacterTextSplitter:
    """Splitter configured from settings.

    The first separator is a regex matching numbered headings, so chunks cut at section
    boundaries and fall back to paragraphs only when a section exceeds ``CHUNK_SIZE``.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=settings.SPLIT_SEPARATORS,
        is_separator_regex=settings.SPLIT_IS_REGEX,
    )


def build_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=settings.EMBEDDING_MODEL)


def stamp_build_config(chunks: list[Document]) -> list[Document]:
    """Record the settings that produced these vectors.

    Lets ``retriever.py`` detect that ``settings.py`` has drifted from what the store was
    actually built with — an ``embedding_model`` mismatch is otherwise silent.
    """
    for chunk in chunks:
        chunk.metadata.update(
            embedding_model=settings.EMBEDDING_MODEL,
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )
    return chunks


def _assert_contract(chunks: list[Document], product_id: str) -> None:
    """Fail before writing if any chunk is missing a contract field.

    Metadata has to survive the loader, ``split_documents``, and JSONB round-tripping; a
    field lost anywhere in that path would otherwise surface much later as an unrenderable
    citation.
    """
    for chunk in chunks:
        missing = FULL_META - chunk.metadata.keys()
        if missing:
            raise IngestionError(
                f"{product_id}: chunk from {chunk.metadata.get('source', '?')} is missing "
                f"metadata fields {sorted(missing)} — refusing to write an incomplete store"
            )


# -- ingestion -------------------------------------------------------------


def ingest_product(
    product_id: str,
    registry: Registry,
    ingested_at: str,
    splitter: RecursiveCharacterTextSplitter,
    embeddings: OpenAIEmbeddings,
    *,
    dry_run: bool = False,
) -> int:
    """Load → chunk → stamp → embed → write one product's collection.

    Idempotent: ``pre_delete_collection`` replaces the collection rather than appending, so
    re-running never duplicates chunks. Returns the number of chunks written.
    """
    docs = load_product_docs(product_id, ingested_at)
    if not docs:
        # A client mid-onboarding may legitimately have only some products documented.
        log.warning("%s: no documents found — skipping", product_id)
        return 0

    chunks = stamp_build_config(splitter.split_documents(docs))
    _assert_contract(chunks, product_id)

    collection = registry.get(product_id).doc_collection
    if dry_run:
        log.info(
            "%s: DRY RUN — %d chunks would be written to '%s'",
            product_id,
            len(chunks),
            collection,
        )
        return len(chunks)

    PGVector.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection,  # from the registry — never hardcoded (I-8)
        connection=settings.DATABASE_URL,
        pre_delete_collection=True,  # full-replace re-ingest
    )
    log.info("%s: %d chunks → '%s'", product_id, len(chunks), collection)
    return len(chunks)


def ingest_all(
    registry: Registry | None = None,
    product_ids: list[str] | None = None,
    *,
    dry_run: bool = False,
) -> dict[str, int]:
    """Ingest every registered product (or the subset in ``product_ids``).

    Returns ``{product_id: chunks_written}``.

    Raises ``IngestionError`` if the run wrote nothing at all — a job that produces an
    empty store is broken, not a no-op. A *single* empty product is only a warning.
    """
    registry = registry or get_registry()
    targets = product_ids or registry.product_ids()
    for pid in targets:
        registry.get(pid)  # raises on unknown ids before any expensive work

    # One timestamp for the whole run, so every chunk it writes is identifiable as its
    # output. UTC and timezone-aware: this job may run on a server in another zone, and a
    # naive local timestamp leaves the audit trail ambiguous.
    ingested_at = datetime.now(UTC).isoformat(timespec="seconds")
    splitter = build_splitter()
    embeddings = build_embeddings()

    log.info(
        "Ingesting %d product(s) — model=%s chunk_size=%d overlap=%d run=%s%s",
        len(targets),
        settings.EMBEDDING_MODEL,
        settings.CHUNK_SIZE,
        settings.CHUNK_OVERLAP,
        ingested_at,
        " [DRY RUN]" if dry_run else "",
    )

    written = {
        pid: ingest_product(pid, registry, ingested_at, splitter, embeddings, dry_run=dry_run)
        for pid in targets
    }

    if not sum(written.values()):
        raise IngestionError(
            f"No documents found for any of {targets} under {settings.DOCS_ROOT}. "
            f"Expected files at <DOCS_ROOT>/<product_id>/ with one of {sorted(LOADERS)}."
        )

    empty = [pid for pid, n in written.items() if not n]
    if empty:
        log.warning("Completed with no documents for: %s", ", ".join(empty))
    log.info(
        "Done — %d chunks across %d product(s)",
        sum(written.values()),
        len(written) - len(empty),
    )
    return written


# -- CLI -------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m app.retrieval.ingestion",
        description="Offline job: ingest product documentation into pgvector.",
    )
    parser.add_argument(
        "--product",
        action="append",
        dest="products",
        metavar="PRODUCT_ID",
        help="Ingest only this product (repeatable). Default: every registered product.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and chunk, but make no embedding calls and write nothing.",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Debug-level logging.")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else settings.LOG_LEVEL,
        format="%(levelname)-8s %(message)s",
    )

    try:
        ingest_all(product_ids=args.products, dry_run=args.dry_run)
    except (IngestionError, KeyError) as exc:
        log.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
