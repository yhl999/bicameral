from __future__ import annotations

import asyncio
import importlib
import sqlite3
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from mcp_server.src.services.change_ledger import ChangeLedger
from mcp_server.src.services.typed_retrieval import TypedRetrievalService
from truth import candidates as candidates_store

promotion_policy_v3 = importlib.import_module("truth.promotion_policy_v3")


class _FakeResult:
    def __init__(self, row: dict[str, Any] | None, nodes_created: int = 0) -> None:
        self._row = row
        self._summary = SimpleNamespace(counters=SimpleNamespace(nodes_created=nodes_created))

    def single(self) -> dict[str, Any] | None:
        return self._row

    def consume(self) -> Any:
        return self._summary


class _FakeSession:
    def __init__(self, *, nodes_created: int) -> None:
        self.nodes_created = nodes_created
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __enter__(self) -> _FakeSession:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
        return None

    def run(self, query: str, params: dict[str, Any]) -> _FakeResult:
        self.calls.append((query, params))
        if "MERGE (c:CoreMemory" in query:
            return _FakeResult(
                {
                    "core_memory_id": params["core_memory_id"],
                    "promoted_at": params["promoted_at"],
                    "candidate_id": params["candidate_id"],
                },
                nodes_created=self.nodes_created,
            )
        return _FakeResult({"rel_count": 1})


class _FakeDriver:
    def __init__(self, *, nodes_created: int) -> None:
        self.nodes_created = nodes_created
        self.last_session: _FakeSession | None = None

    def session(self, database: str | None = None) -> _FakeSession:
        self.last_session = _FakeSession(nodes_created=self.nodes_created)
        return self.last_session


@dataclass
class _ClosableSentinel:
    closed: bool = False

    def close(self) -> None:
        self.closed = True




class _FakeEvidenceRegistry:
    async def resolve_many(self, refs, *, object_ids_by_uri=None, max_items=None):
        return []


def _memory_ledger() -> ChangeLedger:
    conn = sqlite3.connect(':memory:')
    conn.row_factory = sqlite3.Row
    return ChangeLedger(conn)

def _verification(candidate_id: str = "cand-1") -> Any:
    return promotion_policy_v3.VerificationRecord(
        candidate_id=candidate_id,
        verification_status="corroborated",
        evidence_source_ids=["m1", "m2"],
        verifier_version="vtest",
        verified_at="2026-01-01T00:00:00Z",
    )


@pytest.mark.parametrize("nodes_created,expected_created", [(1, True), (0, False)])
def test_promote_candidate_created_is_deterministic_from_merge_counters(
    nodes_created: int,
    expected_created: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Runtime-only write-path test: disable v3 gate so candidates DB is not required.
    monkeypatch.setenv("GRAPHITI_POLICY_V3_ENABLED", "0")
    driver = _FakeDriver(nodes_created=nodes_created)

    result = promotion_policy_v3.promote_candidate(
        candidate_id="cand-1",
        verification=_verification("cand-1"),
        hard_block_check=lambda _: False,
        neo4j_driver=driver,
    )

    assert result["promoted"] is True
    assert result["created"] is expected_created
    assert result["supports_core_edges_attempted"] == 2


def test_promote_candidate_omnode_not_found_skips_gracefully(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """promote_candidate must return promoted=False/reason=omnode_not_found (not raise)
    when the OMNode is absent from Neo4j.  Covers verification-only promotion
    paths where the verification record exists in candidates.db but the source
    OMNode has been evicted or was never written to the graph.
    """

    # Runtime-only write-path test: disable v3 gate so candidates DB is not required.
    monkeypatch.setenv("GRAPHITI_POLICY_V3_ENABLED", "0")

    class _SessionNoOMNode(_FakeSession):
        def __init__(self) -> None:
            super().__init__(nodes_created=0)

        def run(self, query: str, params: dict[str, Any]) -> _FakeResult:
            self.calls.append((query, params))
            if "MERGE (c:CoreMemory" in query:
                # No ledger-backed typed object exists on this path, so the query
                # must still require the OMNode and fail closed when it's gone.
                return _FakeResult(None, nodes_created=0)
            return _FakeResult({"rel_count": 1})

    class _DriverNoOMNode(_FakeDriver):
        def __init__(self) -> None:
            super().__init__(nodes_created=0)

        def session(self, database: str | None = None) -> _SessionNoOMNode:
            self.last_session = _SessionNoOMNode()
            return self.last_session

    result = promotion_policy_v3.promote_candidate(
        candidate_id="ghost-cand-1",
        verification=_verification("ghost-cand-1"),
        hard_block_check=lambda _: False,
        neo4j_driver=_DriverNoOMNode(),
    )

    assert result["promoted"] is False
    assert result["reason"] == "omnode_not_found"
    assert result["candidate_id"] == "ghost-cand-1"
    assert "core_memory_id" in result



def test_promote_candidate_with_ledger_materializes_corememory_without_omnode():
    conn = candidates_store.connect(':memory:')
    ledger = _memory_ledger()

    candidate = candidates_store.upsert_candidate(
        conn,
        subject='user:principal',
        predicate='pref.travel_style',
        scope='private',
        assertion_type='preference',
        value={'value': 'carry-on only'},
        evidence_refs=[
            {
                'source_key': 'om:s1_observational_memory:node:travel-style',
                'evidence_id': 'm1',
                'scope': 's1_observational_memory',
            }
        ],
        speaker_id='owner',
        confidence=0.92,
        origin='extracted',
    )

    class _LedgerBackedNoOMNodeSession(_FakeSession):
        def __init__(self) -> None:
            super().__init__(nodes_created=1)

        def run(self, query: str, params: dict[str, Any]) -> _FakeResult:
            self.calls.append((query, params))
            if "MERGE (c:CoreMemory" in query:
                assert "OPTIONAL MATCH (n:OMNode {node_id:$candidate_id})" in query
                return _FakeResult(
                    {
                        "core_memory_id": params["core_memory_id"],
                        "promoted_at": params["promoted_at"],
                        "candidate_id": params["candidate_id"],
                    },
                    nodes_created=1,
                )
            return _FakeResult({"rel_count": 1})

    class _LedgerBackedNoOMNodeDriver(_FakeDriver):
        def __init__(self) -> None:
            super().__init__(nodes_created=1)

        def session(self, database: str | None = None) -> _LedgerBackedNoOMNodeSession:
            self.last_session = _LedgerBackedNoOMNodeSession()
            return self.last_session

    driver = _LedgerBackedNoOMNodeDriver()
    result = promotion_policy_v3.promote_candidate(
        candidate_id=candidate.candidate_id,
        verification=_verification(candidate.candidate_id),
        hard_block_check=lambda _: False,
        neo4j_driver=driver,
        candidates_conn=conn,
        ledger=ledger,
    )

    assert result['promoted'] is True
    assert result['ledger_root_id'] is not None
    assert driver.last_session is not None
    core_query, core_params = driver.last_session.calls[0]
    assert "OPTIONAL MATCH (n:OMNode {node_id:$candidate_id})" in core_query
    assert "c.materialization_source = CASE" in core_query
    assert core_params['content'] == 'user:principal pref.travel_style {"value": "carry-on only"}'


def test_promote_candidate_with_candidates_conn_requires_ledger(monkeypatch: pytest.MonkeyPatch) -> None:
    conn = candidates_store.connect(':memory:')
    candidate = candidates_store.upsert_candidate(
        conn,
        subject='user:principal',
        predicate='pref.food',
        scope='private',
        assertion_type='preference',
        value={'value': 'ramen'},
        evidence_refs=[{'source_key': 'om:s1_observational_memory:node:food', 'evidence_id': 'm1', 'scope': 's1_observational_memory'}],
        speaker_id='owner',
        confidence=0.92,
        origin='extracted',
    )

    result = promotion_policy_v3.promote_candidate(
        candidate_id=candidate.candidate_id,
        verification=_verification(candidate.candidate_id),
        hard_block_check=lambda _: False,
        neo4j_driver=_FakeDriver(nodes_created=1),
        candidates_conn=conn,
        ledger=None,
    )

    assert result['promoted'] is False
    assert result['reason'] == 'ledger_missing'
    row = conn.execute('SELECT status, ledger_event_id FROM candidates WHERE candidate_id = ?', (candidate.candidate_id,)).fetchone()
    assert row['status'] == 'auto_promoted'
    assert row['ledger_event_id'] is None


def test_shared_driver_is_lazy_singleton(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = _ClosableSentinel()
    load_count = 0

    def _fake_loader() -> _ClosableSentinel:
        nonlocal load_count
        load_count += 1
        return sentinel

    monkeypatch.setattr(promotion_policy_v3, "_neo4j_driver_from_env", _fake_loader)
    monkeypatch.setattr(promotion_policy_v3, "_NEO4J_DRIVER_SINGLETON", None)

    first = promotion_policy_v3._shared_neo4j_driver()
    second = promotion_policy_v3._shared_neo4j_driver()

    assert first is sentinel
    assert second is sentinel
    assert load_count == 1

    promotion_policy_v3._close_shared_neo4j_driver()
    assert sentinel.closed is True


def test_promote_candidate_with_candidates_conn_writes_ledger_first_and_supports_typed_retrieval():
    conn = candidates_store.connect(':memory:')
    ledger = _memory_ledger()

    candidate = candidates_store.upsert_candidate(
        conn,
        subject='user:principal',
        predicate='pref.travel_style',
        scope='private',
        assertion_type='preference',
        value={'value': 'carry-on only'},
        evidence_refs=[
            {
                'source_key': 'om:s1_observational_memory:node:travel-style',
                'evidence_id': 'm1',
                'scope': 's1_observational_memory',
            }
        ],
        speaker_id='owner',
        confidence=0.92,
        origin='extracted',
    )
    assert candidate.status == 'auto_promoted'

    driver = _FakeDriver(nodes_created=1)
    result = promotion_policy_v3.promote_candidate(
        candidate_id=candidate.candidate_id,
        verification=_verification(candidate.candidate_id),
        hard_block_check=lambda _: False,
        neo4j_driver=driver,
        candidates_conn=conn,
        ledger=ledger,
    )

    assert result['promoted'] is True
    assert result['ledger_event_id'] is not None
    assert result['ledger_object_id'] is not None
    assert result['ledger_root_id'] is not None
    assert result['core_memory_id'] == promotion_policy_v3.sha256_hex(f"core|{result['ledger_root_id']}")

    row = conn.execute(
        'SELECT status, ledger_event_id FROM candidates WHERE candidate_id = ?',
        (candidate.candidate_id,),
    ).fetchone()
    assert row['status'] == 'approved'
    assert row['ledger_event_id'] == result['ledger_event_id']

    promoted = ledger.current_state_facts()[0]
    assert promoted.candidate_id == candidate.candidate_id
    assert promoted.source_lane == 's1_observational_memory'
    assert promoted.promotion_status == 'promoted'

    service = TypedRetrievalService(ledger=ledger, evidence_registry=_FakeEvidenceRegistry())
    response = asyncio.run(
        service.search(
            query='travel carry-on',
            object_types=['state'],
            metadata_filters={'source_lane': {'eq': 's1_observational_memory'}},
            max_results=5,
        )
    )

    assert response['counts']['state'] == 1
    assert response['state'][0]['object_id'] == result['ledger_object_id']
    assert response['state'][0]['source_lane'] == 's1_observational_memory'
    assert response['limits_applied']['materialization']['om_projection']['enabled'] is False

    assert driver.last_session is not None
    core_query, core_params = driver.last_session.calls[0]
    assert 'MERGE (c:CoreMemory' in core_query
    assert core_params['ledger_event_id'] == result['ledger_event_id']
    assert core_params['ledger_root_id'] == result['ledger_root_id']
    assert core_params['source_lane'] == 's1_observational_memory'


def test_promote_candidate_supersede_reuses_ledger_root_for_corememory_mapping():
    conn = candidates_store.connect(':memory:')
    ledger = _memory_ledger()
    driver = _FakeDriver(nodes_created=1)

    first = candidates_store.upsert_candidate(
        conn,
        subject='user:principal',
        predicate='pref.editor',
        scope='private',
        assertion_type='preference',
        value={'value': 'vim'},
        evidence_refs=[
            {'source_key': 'om:s1_observational_memory:node:editor-vim', 'evidence_id': 'm1', 'scope': 's1_observational_memory'}
        ],
        speaker_id='owner',
        confidence=0.92,
        origin='extracted',
    )
    first_result = promotion_policy_v3.promote_candidate(
        candidate_id=first.candidate_id,
        verification=_verification(first.candidate_id),
        hard_block_check=lambda _: False,
        neo4j_driver=driver,
        candidates_conn=conn,
        ledger=ledger,
    )
    first_fact = ledger.current_state_facts()[0]

    second = candidates_store.upsert_candidate(
        conn,
        subject='user:principal',
        predicate='pref.editor',
        scope='private',
        assertion_type='preference',
        value={'value': 'helix'},
        evidence_refs=[
            {'source_key': 'om:s1_observational_memory:node:editor-helix', 'evidence_id': 'm2', 'scope': 's1_observational_memory'}
        ],
        speaker_id='owner',
        confidence=0.92,
        origin='extracted',
        conflict_with_fact_id=first_fact.object_id,
        seeded_supersede_ok=True,
        explicit_update=True,
    )
    assert second.status == 'auto_supersede'

    second_result = promotion_policy_v3.promote_candidate(
        candidate_id=second.candidate_id,
        verification=_verification(second.candidate_id),
        hard_block_check=lambda _: False,
        neo4j_driver=driver,
        candidates_conn=conn,
        ledger=ledger,
    )

    assert second_result['promoted'] is True
    assert second_result['ledger_root_id'] == first_result['ledger_root_id']
    assert second_result['core_memory_id'] == first_result['core_memory_id']

    current = ledger.current_state_facts()[0]
    assert current.value == {'value': 'helix'}
    assert current.root_id == first_fact.root_id
    assert current.version == 2
