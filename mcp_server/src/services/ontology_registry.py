"""Ontology registry for lane-specific extraction.

Loads per-group ontology profiles from a YAML config file and provides
a simple lookup interface. Adding a new extraction lane requires only
adding a YAML block — zero code changes.

Usage:
    registry = OntologyRegistry.load("config/extraction_ontologies.yaml")
    profile = registry.get("s1_inspiration_short_form")
    # profile.entity_types  -> dict[str, type[BaseModel]]
    # profile.extraction_emphasis -> str
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

logger = logging.getLogger(__name__)

_SAFE_TYPE_NAME_RE = re.compile(r'^[A-Z][A-Za-z0-9_]{0,63}$')


@dataclass(frozen=True)
class OntologyProfile:
    """Immutable extraction profile for a single graph lane.

    Attributes:
        entity_types: Mapping of entity type name → dynamically created
            Pydantic model class (same format as graphiti-core expects).
        relationship_types: List of relationship type dicts (name + description),
            stored for documentation and future use in constrained extraction.
        extraction_emphasis: Prompt hint injected into the LLM extraction
            call to steer focus toward lane-relevant patterns.
    """

    entity_types: dict[str, type[BaseModel]] = field(default_factory=dict)
    relationship_types: list[dict[str, str]] = field(default_factory=list)
    extraction_emphasis: str = ""


def _build_entity_types(raw_types: list[dict[str, str]]) -> dict[str, type[BaseModel]]:
    """Convert YAML entity type definitions into Pydantic model classes.

    Each entry becomes a dynamically created BaseModel subclass whose
    ``__doc__`` carries the description — matching the format that
    graphiti-core's ``add_episode(entity_types=...)`` expects.
    """
    result: dict[str, type[BaseModel]] = {}
    for entry in raw_types:
        name = entry["name"]
        if not _SAFE_TYPE_NAME_RE.match(name):
            raise ValueError(
                f"Invalid entity type name {name!r} — must match {_SAFE_TYPE_NAME_RE.pattern}"
            )
        description = entry.get("description", "")
        model = type(name, (BaseModel,), {"__doc__": description, "__module__": __name__})
        result[name] = model
    return result


class OntologyRegistry:
    """Registry mapping ``group_id`` → :class:`OntologyProfile`.

    Loaded once at startup from a YAML file and cached for the lifetime
    of the process. Groups not listed in the config fall back to a
    caller-supplied default (typically the global ``entity_types`` from
    ``config.yaml``).
    """

    def __init__(self, profiles: dict[str, OntologyProfile]) -> None:
        self._profiles = profiles

    # ── Factory ────────────────────────────────────────────────

    @classmethod
    def load(cls, path: str | Path) -> OntologyRegistry:
        """Load ontology profiles from a YAML config file.

        Args:
            path: Filesystem path to ``extraction_ontologies.yaml``.

        Returns:
            A populated :class:`OntologyRegistry`.

        Raises:
            FileNotFoundError: If *path* does not exist.
            yaml.YAMLError: If the file is not valid YAML.
        """
        path = Path(path)
        logger.info("Loading extraction ontologies from %s", path)

        raw: dict[str, Any] = yaml.safe_load(path.read_text()) or {}

        profiles: dict[str, OntologyProfile] = {}
        for group_id, definition in raw.items():
            # Allow top-level metadata keys (ex: schema_version, om_extractor)
            # without treating them as ontology lane profiles.
            if not isinstance(definition, dict) or 'entity_types' not in definition:
                logger.debug("Skipping non-ontology key: %s", group_id)
                continue
            entity_types = _build_entity_types(definition.get("entity_types", []))
            relationship_types = definition.get("relationship_types", [])
            extraction_emphasis = definition.get("extraction_emphasis", "")

            profiles[group_id] = OntologyProfile(
                entity_types=entity_types,
                relationship_types=relationship_types,
                extraction_emphasis=extraction_emphasis,
            )
            logger.info(
                "  Loaded ontology for %s: %d entity types, %d relationship types",
                group_id,
                len(entity_types),
                len(relationship_types),
            )

        logger.info("Ontology registry loaded: %d lanes configured", len(profiles))
        return cls(profiles)

    # ── Lookup ─────────────────────────────────────────────────

    def get(self, group_id: str) -> OntologyProfile | None:
        """Return the ontology profile for *group_id*, or ``None`` if not configured.

        Callers should fall back to their default entity types when ``None``
        is returned — this keeps groups without explicit config on the
        existing generic ontology (backwards-compatible).
        """
        return self._profiles.get(group_id)

    def has(self, group_id: str) -> bool:
        """Check whether *group_id* has an explicit ontology profile."""
        return group_id in self._profiles

    @property
    def configured_groups(self) -> list[str]:
        """Return the list of group_ids with explicit ontology profiles."""
        return list(self._profiles.keys())
