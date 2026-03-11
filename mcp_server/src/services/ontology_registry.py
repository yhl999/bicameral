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

import copy
import logging
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

if __name__ == 'mcp_server.src.services.ontology_registry':
    sys.modules.setdefault('services.ontology_registry', sys.modules[__name__])
elif __name__ == 'services.ontology_registry':
    sys.modules.setdefault('mcp_server.src.services.ontology_registry', sys.modules[__name__])

logger = logging.getLogger(__name__)

_SAFE_TYPE_NAME_RE = re.compile(r'^[A-Z][A-Za-z0-9_]{0,63}$')
_KNOWN_METADATA_KEYS = {"om_extractor"}
_VALID_EXTRACTION_MODES = {"permissive", "constrained_soft"}

# Hard cap on intent_guidance / extraction_emphasis length before prompt injection.
# Prevents accidental config bloat from consuming excessive LLM context window tokens.
# Tighter than runtime truncation — enforced at load time so callers see bounded values.
_INTENT_GUIDANCE_MAX_CHARS: int = 2048

# Non-printable control characters that should never appear in prompt-injected config
# values.  Keeps standard whitespace (\t, \n, \r) which are valid in multi-line YAML.
_CONTROL_CHAR_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')


def _sanitize_intent_guidance(text: str, field_name: str, lane_id: str) -> str:
    """Sanitize and bound-check a prompt-injected config string.

    Applies three transforms, in order:
    1. Strip non-printable control characters (keeps \\t, \\n, \\r).
    2. Strip leading/trailing whitespace.
    3. Enforce :data:`_INTENT_GUIDANCE_MAX_CHARS` hard length cap.

    Args:
        text: Raw field value from YAML.
        field_name: Name of the field (for log messages).
        lane_id: Lane group_id (for log messages).

    Returns:
        Sanitized string, safe for prompt injection.
    """
    # Strip dangerous non-printable control characters.
    sanitized = _CONTROL_CHAR_RE.sub('', text)
    if sanitized != text:
        logger.warning(
            "Stripped non-printable control characters from %s for lane %s. "
            "Check the YAML config for embedded escape sequences.",
            field_name, lane_id,
        )
    sanitized = sanitized.strip()
    # Hard length cap.
    if len(sanitized) > _INTENT_GUIDANCE_MAX_CHARS:
        logger.warning(
            "%s for lane %s exceeds %d chars (%d); truncating. "
            "Shorten the YAML value to suppress this warning.",
            field_name, lane_id, _INTENT_GUIDANCE_MAX_CHARS, len(sanitized),
        )
        sanitized = sanitized[:_INTENT_GUIDANCE_MAX_CHARS]
    return sanitized


def _merge_ontology_documents(base: Any, overlay: Any) -> Any:
    """Merge two ontology YAML fragments deterministically.

    Rules:
    - mappings merge recursively
    - lists replace wholesale
    - scalars replace wholesale

    This keeps overlay behaviour explicit and predictable: an overlay may add new
    lanes, override selected scalar/dict fields, or replace an entire lane block
    by re-declaring it.
    """
    if base is None:
        return copy.deepcopy(overlay)
    if overlay is None:
        return copy.deepcopy(base)
    if isinstance(base, dict) and isinstance(overlay, dict):
        merged = {key: copy.deepcopy(value) for key, value in base.items()}
        for key, value in overlay.items():
            if key in merged:
                merged[key] = _merge_ontology_documents(merged[key], value)
            else:
                merged[key] = copy.deepcopy(value)
        return merged
    return copy.deepcopy(overlay)


@dataclass(frozen=True)
class OntologyProfile:
    """Immutable extraction profile for a single graph lane.

    Attributes:
        entity_types: Mapping of entity type name → dynamically created
            Pydantic model class (same format as graphiti-core expects).
        relationship_types: List of relationship type dicts (name + description),
            stored for documentation / reference.
        edge_types: Mapping of relationship type name → dynamically created
            Pydantic model class (same format as graphiti-core expects for
            ``add_episode(edge_types=...)``).  Built from ``relationship_types``
            at load time.
        extraction_emphasis: Prompt hint injected into the LLM extraction
            call to steer focus toward lane-relevant patterns.
            Deprecated alias: use ``intent_guidance`` in new YAML configs.
        intent_guidance: Per-lane natural-language description of what the
            extraction should focus on.  Passed as
            ``custom_extraction_instructions`` to Graphiti Core.  If not set,
            falls back to ``extraction_emphasis`` for backward compatibility.
        extraction_mode: Controls extraction strictness for this lane.
            - ``'permissive'`` (default): extract broadly; all relationship
              types and entity types are allowed.
            - ``'constrained_soft'``: ontology-conformant mode.  Uses
              dedicated prompt branches and code-level enforcement to reduce
              noise and increase conformance to the defined ontology.
    """

    entity_types: dict[str, type[BaseModel]] = field(default_factory=dict)
    relationship_types: list[dict[str, str]] = field(default_factory=list)
    edge_types: dict[str, type[BaseModel]] = field(default_factory=dict)
    extraction_emphasis: str = ""
    intent_guidance: str = ""
    extraction_mode: str = "permissive"


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


def _build_edge_types(raw_types: list[dict[str, str]]) -> dict[str, type[BaseModel]]:
    """Convert YAML relationship type definitions into Pydantic model classes.

    Mirror of :func:`_build_entity_types` for relationship types.  Each entry
    becomes a dynamically created BaseModel subclass whose ``__doc__`` carries
    the description — matching the format that graphiti-core's
    ``add_episode(edge_types=...)`` expects.

    Relationship type names are typically ALL_CAPS with underscores (e.g.
    ``USES_MOVE``), which is valid for Python class names.
    """
    result: dict[str, type[BaseModel]] = {}
    for entry in raw_types:
        name = entry["name"]
        if not _SAFE_TYPE_NAME_RE.match(name):
            raise ValueError(
                f"Invalid relationship type name {name!r} — must match {_SAFE_TYPE_NAME_RE.pattern}"
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

    @staticmethod
    def _load_yaml_document(source_path: Path) -> dict[str, Any]:
        """Load a single ontology YAML file into a dict document."""
        source_raw = yaml.safe_load(source_path.read_text()) or {}
        if not isinstance(source_raw, dict):
            raise ValueError(
                f"Ontology config {source_path} must deserialize to a mapping, got {type(source_raw).__name__}"
            )
        return source_raw

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        overlay_paths: Sequence[str | Path] | None = None,
    ) -> OntologyRegistry:
        """Load ontology profiles from a YAML config file plus optional overlays.

        Composition contract:
        - ``path`` is loaded first as the base ontology document.
        - ``overlay_paths`` are applied in order.
        - mappings merge recursively.
        - lists and scalars replace wholesale.

        This keeps the shared/base ontology stable while allowing private or
        deployment-local overlays to add lanes or override specific lane blocks.

        Args:
            path: Filesystem path to the base ``extraction_ontologies.yaml``.
            overlay_paths: Optional YAML fragments layered on top of *path*.

        Returns:
            A populated :class:`OntologyRegistry`.

        Raises:
            FileNotFoundError: If the base path does not exist.
            yaml.YAMLError: If the base file does not contain valid YAML.
            ValueError: If the base document is not a mapping.

        Overlay documents are optional. If an overlay path is missing or invalid,
        this method emits a warning and continues with the remaining overlays,
        preserving the base ontology behavior.
        """
        base_path = Path(path)
        overlay_paths = [Path(p) for p in (overlay_paths or ())]
        paths = [base_path, *overlay_paths]
        logger.info("Loading extraction ontologies from %s", ", ".join(str(p) for p in paths))

        raw: dict[str, Any] = {}
        for idx, source_path in enumerate(paths):
            try:
                source_raw = cls._load_yaml_document(source_path)
            except (FileNotFoundError, OSError, ValueError, yaml.YAMLError) as error:
                if idx == 0:
                    raise
                logger.warning(
                    "Skipping ontology overlay %s due load failure: %s",
                    source_path,
                    error,
                )
                continue

            raw = _merge_ontology_documents(raw, source_raw)

        profiles: dict[str, OntologyProfile] = {}
        for group_id, definition in raw.items():
            if not isinstance(definition, dict):
                logger.debug("Skipping non-dict ontology key: %s", group_id)
                continue
            if "entity_types" not in definition:
                if group_id not in _KNOWN_METADATA_KEYS:
                    logger.warning(
                        "Skipping dict ontology key without entity_types: %s", group_id
                    )
                continue
            entity_types = _build_entity_types(definition.get("entity_types", []))
            relationship_types = definition.get("relationship_types", [])
            edge_types = _build_edge_types(relationship_types)
            # Sanitize string fields before they can reach LLM prompts.
            # Both fields are treated as UNTRUSTED operator config: bounded length
            # + control-char scrubbing applied at load time, once.
            extraction_emphasis_raw = definition.get("extraction_emphasis", "")
            extraction_emphasis = _sanitize_intent_guidance(
                extraction_emphasis_raw, "extraction_emphasis", group_id
            )
            # intent_guidance is the canonical key for new configs.
            # Falls back to extraction_emphasis for backward compatibility.
            intent_guidance_raw = definition.get("intent_guidance", extraction_emphasis_raw)
            intent_guidance = _sanitize_intent_guidance(
                intent_guidance_raw, "intent_guidance", group_id
            )
            extraction_mode = definition.get("extraction_mode", "permissive")
            if extraction_mode not in _VALID_EXTRACTION_MODES:
                logger.warning(
                    "Invalid extraction_mode %r for %s — falling back to 'permissive'. "
                    "Valid values: %s",
                    extraction_mode,
                    group_id,
                    _VALID_EXTRACTION_MODES,
                )
                extraction_mode = "permissive"

            profiles[group_id] = OntologyProfile(
                entity_types=entity_types,
                relationship_types=relationship_types,
                edge_types=edge_types,
                extraction_emphasis=extraction_emphasis,
                intent_guidance=intent_guidance,
                extraction_mode=extraction_mode,
            )
            logger.info(
                "  Loaded ontology for %s: %d entity types, %d relationship types, mode=%s",
                group_id,
                len(entity_types),
                len(relationship_types),
                extraction_mode,
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
