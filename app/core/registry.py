"""Product registry — loads and validates ``app/config/products.yaml``.

Loaded once at startup. Every product-specific fact in the system resolves through
here, so no node or prompt ever hardcodes a product name (invariant I-8).

Consumers:
    classify        label set from ``product_ids()``
    retrieve_docs   ``get(pid).doc_collection``
    retrieve_db     ``get(pid).sql_templates`` (filter) and ``db_tables`` (allowlist)
    decline         ``display_names()`` for the off_topic template
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

# Default location, relative to the repo root (app/core/registry.py -> app/config/).
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "products.yaml"


class RegistryError(Exception):
    """Raised when products.yaml is missing, malformed, or fails validation.

    Always fatal: a bad registry means prompts and allowlists would be built from
    unverified config, so startup fails loudly rather than degrading silently.
    """


class ProductConfig(BaseModel):
    """One product's configuration block."""

    # Reject unknown keys — a typo'd field name (e.g. `doc_collections`) would
    # otherwise be silently dropped and surface much later as an empty retrieval.
    model_config = ConfigDict(extra="forbid", frozen=True)

    display_name: str = Field(min_length=1)
    doc_collection: str = Field(min_length=1)
    db_tables: list[str] = Field(min_length=1)
    sql_templates: list[str] = Field(min_length=1)

    @field_validator("db_tables", "sql_templates")
    @classmethod
    def _no_blank_or_duplicate_entries(cls, v: list[str], info) -> list[str]:
        if any(not item or not item.strip() for item in v):
            raise ValueError(f"{info.field_name} contains a blank entry")
        if len(set(v)) != len(v):
            raise ValueError(f"{info.field_name} contains duplicates")
        return v


class Registry:
    """Immutable view over the validated product configuration."""

    def __init__(self, path: str | Path = DEFAULT_CONFIG_PATH) -> None:
        self.path = Path(path)
        raw = self._read(self.path)
        self.products: dict[str, ProductConfig] = self._parse(raw, self.path)
        self._check_cross_product_uniqueness()

    # -- loading -----------------------------------------------------------

    @staticmethod
    def _read(path: Path) -> dict:
        try:
            with path.open(encoding="utf-8") as fh:
                raw = yaml.safe_load(fh)
        except FileNotFoundError as exc:
            raise RegistryError(f"products.yaml not found at {path}") from exc
        except yaml.YAMLError as exc:
            raise RegistryError(f"products.yaml at {path} is not valid YAML: {exc}") from exc

        if not isinstance(raw, dict) or not raw:
            raise RegistryError(
                f"products.yaml at {path} must be a non-empty mapping of "
                f"product_id -> config, got {type(raw).__name__}"
            )
        return raw

    @staticmethod
    def _parse(raw: dict, path: Path) -> dict[str, ProductConfig]:
        products: dict[str, ProductConfig] = {}
        for product_id, block in raw.items():
            if not isinstance(block, dict):
                raise RegistryError(
                    f"product '{product_id}' in {path} must be a mapping, "
                    f"got {type(block).__name__}"
                )
            try:
                products[product_id] = ProductConfig(**block)
            except Exception as exc:  # pydantic ValidationError or bad kwargs
                raise RegistryError(f"product '{product_id}' in {path} is invalid: {exc}") from exc
        return products

    def _check_cross_product_uniqueness(self) -> None:
        """Two products sharing a doc_collection would cross-contaminate retrieval;
        sharing a template name would make executor filtering ambiguous."""
        for field in ("doc_collection", "sql_templates"):
            seen: dict[str, str] = {}
            for pid, cfg in self.products.items():
                value = getattr(cfg, field)
                for item in [value] if isinstance(value, str) else value:
                    if item in seen:
                        raise RegistryError(
                            f"{field} '{item}' is used by both '{seen[item]}' and "
                            f"'{pid}' — must be unique per product"
                        )
                    seen[item] = pid

    # -- accessors ---------------------------------------------------------

    def get(self, product_id: str) -> ProductConfig:
        """Look up one product. Raises on unknown IDs — never returns a default,
        so an LLM-invented product label cannot silently resolve to something."""
        try:
            return self.products[product_id]
        except KeyError:
            raise KeyError(
                f"unknown product '{product_id}'; known products: {self.product_ids()}"
            ) from None

    def product_ids(self) -> list[str]:
        return list(self.products)

    def display_names(self) -> list[str]:
        """Customer-facing product names, for the off_topic decline template."""
        return [cfg.display_name for cfg in self.products.values()]

    def is_valid_product(self, product_id: str) -> bool:
        """Used by classify to reject invented product labels before they flow on."""
        return product_id in self.products

    def doc_collections(self) -> dict[str, str]:
        return {pid: cfg.doc_collection for pid, cfg in self.products.items()}

    def all_db_tables(self) -> set[str]:
        """Union of every product's tables — the global executor table allowlist."""
        return {t for cfg in self.products.values() for t in cfg.db_tables}

    def templates_for(self, product_ids: list[str]) -> list[str]:
        """Template names visible to retrieve_db for the routed products only."""
        names: list[str] = []
        for pid in product_ids:
            for name in self.get(pid).sql_templates:
                if name not in names:
                    names.append(name)
        return names

    def __len__(self) -> int:
        return len(self.products)

    def __repr__(self) -> str:
        return f"Registry(path={self.path!s}, products={self.product_ids()})"


@cache
def get_registry(path: str | Path = DEFAULT_CONFIG_PATH) -> Registry:
    """Process-wide singleton — config is read from disk once at startup."""
    return Registry(path)
