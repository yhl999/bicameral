"""Response type definitions for Graphiti MCP Server."""

from typing import Any

from typing_extensions import NotRequired, TypedDict


class ErrorResponse(TypedDict):
    error: str
    message: NotRequired[str]
    details: NotRequired[Any]


class SuccessResponse(TypedDict):
    message: str
    details: NotRequired[Any]


class NodeResult(TypedDict):
    uuid: str
    name: str
    labels: list[str]
    created_at: str | None
    summary: str | None
    group_id: str
    attributes: dict[str, Any]


class NodeSearchResponse(TypedDict):
    message: str
    nodes: list[NodeResult]


class FactSearchResponse(TypedDict):
    message: str
    facts: list[dict[str, Any]]


class EpisodeSearchResponse(TypedDict):
    message: str
    episodes: list[dict[str, Any]]


class TypedMemoryQueryMetadata(TypedDict):
    subject: str
    predicate: str | None
    scope: str | None
    group_ids: list[str]
    lane_alias: list[str] | None
    limit: int
    result_count: int
    truncated: bool


class CurrentStateResponse(TypedDict):
    message: str
    facts: list[dict[str, Any]]
    metadata: TypedMemoryQueryMetadata


class HistoryResponse(TypedDict):
    message: str
    history: list[dict[str, Any]]
    metadata: TypedMemoryQueryMetadata


class StatusResponse(TypedDict):
    status: str
    message: str
