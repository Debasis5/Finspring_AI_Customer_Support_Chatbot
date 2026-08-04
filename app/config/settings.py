"""Application settings — env vars, model names, and frozen pipeline parameters.

Single source of truth for everything tunable. Two rules:

* **Model names live here only** (invariant I-9). ``services/llm.py`` reads
  ``CHEAP_MODEL`` / ``STRONG_MODEL`` from this module and nothing else names a model.
* **Ingestion parameters are frozen here**, decided in
  ``notebooks/experiments/1_4_ingestion.ipynb``. Changing ``CHUNK_SIZE``,
  ``CHUNK_OVERLAP``, or ``EMBEDDING_MODEL`` invalidates the existing vector store —
  re-run ingestion afterwards.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Load .env before Settings reads the environment. Real env vars win over the file.
load_dotenv(REPO_ROOT / ".env", override=False)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
        protected_namespaces=(),  # allow field names beginning with "model_"
    )

    # -- Database ----------------------------------------------------------
    # One Postgres instance serves three roles: pgvector store, application
    # tables, and the LangGraph checkpointer. No new infra (CLAUDE.md §1).
    DATABASE_URL: str = Field(
        description="Read-write connection (ingestion, checkpointer). SQLAlchemy URL."
    )
    # Stage 3: the SQL template executor runs on this SELECT-only role, never on
    # DATABASE_URL. Optional until db/schema.sql creates the role.
    DATABASE_URL_READONLY: str | None = None

    # -- OpenAI ------------------------------------------------------------
    OPENAI_API_KEY: str = Field(
        min_length=1, description="Required for embeddings and all LLM calls."
    )

    # Model tiering (CLAUDE.md invariant 9). Configure ONLY here.
    CHEAP_MODEL: str = "gpt-4o-mini"  # classify, verify, retrieve_db template selection
    STRONG_MODEL: str = "gpt-4o"  # generate only
    CLASSIFY_TEMPERATURE: float = 0.0
    GENERATE_TEMPERATURE: float = 0.0
    VERIFY_TEMPERATURE: float = 0.0

    # -- Ingestion / retrieval (frozen in experiments/1_4_ingestion.ipynb) ------
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    CHUNK_SIZE: int = Field(default=1000, gt=0)
    CHUNK_OVERLAP: int = Field(default=150, ge=0)

    # Section-aware splitting. The first separator is a regex matching numbered
    # headings ("\n\n5. Premature withdrawal"), so the splitter prefers to cut at
    # section boundaries and falls back to paragraphs only when a section exceeds
    # CHUNK_SIZE. Measured on all four KB docs: multi-section chunks 9 -> 1, and
    # chunks starting at a heading 6 -> 41 (headings are strong topical signal in
    # the embedding, and give `generate` a titled context unit rather than an
    # orphaned fragment). Degrades gracefully: a KB without numbered headings
    # simply never matches the first separator and splits on paragraphs as before.
    # SPLIT_IS_REGEX must stay True while the first entry is a pattern.
    SPLIT_SEPARATORS: list[str] = [r"\n\n\d+\.\s", "\n\n", "\n", " ", ""]
    SPLIT_IS_REGEX: bool = True
    TOP_K: int = Field(default=4, gt=0, description="Chunks retrieved per product.")

    # -- Graph behaviour ---------------------------------------------------
    MAX_RETRIES: int = Field(
        default=2,
        ge=0,
        description="Shared retry budget for re-entering `generate` — a single counter "
        "across both groundedness and advice-drift failures (CLAUDE.md §2).",
    )
    MAX_INPUT_LENGTH: int = Field(
        default=2000, gt=0, description="pre_checks rejects queries longer than this."
    )
    ENABLE_MODERATION: bool = Field(
        default=False, description="Optional OpenAI moderation call in pre_checks."
    )

    # -- DB executor guardrails (Stage 3) ----------------------------------
    MAX_ROWS: int = Field(default=100, gt=0, description="Row cap on template results.")
    STATEMENT_TIMEOUT_MS: int = Field(default=5000, gt=0)

    # -- Paths -------------------------------------------------------------
    DOCS_ROOT: Path = REPO_ROOT / "data" / "docs"
    PRODUCTS_CONFIG: Path = REPO_ROOT / "app" / "config" / "products.yaml"
    SQL_TEMPLATES_CONFIG: Path = REPO_ROOT / "app" / "config" / "sql_templates.yaml"

    # -- Observability (Stage 6; optional) ---------------------------------
    LANGSMITH_TRACING: bool = False
    LANGSMITH_API_KEY: str | None = None
    LANGSMITH_PROJECT: str = "finspring-chatbot-v1"
    LOG_LEVEL: str = "INFO"

    # -- Validation --------------------------------------------------------

    @field_validator("DATABASE_URL", "DATABASE_URL_READONLY")
    @classmethod
    def _must_be_psycopg_url(cls, v: str | None, info) -> str | None:
        """langchain-postgres and the checkpointer both expect the psycopg (v3)
        driver; a bare `postgresql://` URL silently selects psycopg2."""
        if v is None:
            return v
        if not v.startswith("postgresql+psycopg://"):
            raise ValueError(
                f"{info.field_name} must start with 'postgresql+psycopg://' "
                f"(psycopg 3 driver), got: {v.split('://')[0]}://..."
            )
        return v

    @model_validator(mode="after")
    def _overlap_below_chunk_size(self) -> Settings:
        if self.CHUNK_OVERLAP >= self.CHUNK_SIZE:
            raise ValueError(
                f"CHUNK_OVERLAP ({self.CHUNK_OVERLAP}) must be smaller than "
                f"CHUNK_SIZE ({self.CHUNK_SIZE})"
            )
        return self

    # -- Derived -----------------------------------------------------------

    @property
    def executor_url(self) -> str:
        """Connection the SQL executor runs on.

        Falls back to DATABASE_URL only before Stage 3 creates the read-only role;
        once DATABASE_URL_READONLY is set it always wins (invariant I-2).
        """
        return self.DATABASE_URL_READONLY or self.DATABASE_URL


@cache
def get_settings() -> Settings:
    """Process-wide singleton — env is read once, on first access.

    Deliberately lazy: importing this module must not require a populated .env,
    or unit tests that construct their own Settings could not import it.
    """
    return Settings()


def __getattr__(name: str):
    """Module-level ``settings`` resolved on first attribute access.

    Lets callers write ``from app.config.settings import settings`` while keeping
    validation deferred to that moment rather than import time.
    """
    if name == "settings":
        return get_settings()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
