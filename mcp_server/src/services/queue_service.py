"""Queue service for managing episode processing."""

import asyncio
import logging
import re
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

_SAFE_GROUP_ID_RE = re.compile(r'^[a-zA-Z0-9_]+$')
_MAX_QUEUE_SIZE = 1000


class QueueService:
    """Service for managing sequential episode processing queues by group_id."""

    def __init__(self):
        """Initialize the queue service."""
        # Dictionary to store queues for each group_id
        self._episode_queues: dict[str, asyncio.Queue] = {}
        # Dictionary to track if a worker is running for each group_id
        self._queue_workers: dict[str, bool] = {}
        # Legacy single Graphiti client (fallback)
        self._graphiti_client: Any = None
        # Optional per-group client resolver
        self._graphiti_client_resolver: Callable[[str], Awaitable[Any]] | None = None
        # Optional per-group ontology resolver.
        # New-style: returns (entity_types, extraction_emphasis) tuple.
        # Legacy: returns entity_types dict or None.
        self._ontology_resolver: Callable[[str], tuple[dict | None, str] | dict | None] | None = None

    async def add_episode_task(
        self, group_id: str, process_func: Callable[[], Awaitable[None]]
    ) -> int:
        """Add an episode processing task to the queue.

        Args:
            group_id: The group ID for the episode
            process_func: The async function to process the episode

        Returns:
            The position in the queue
        """
        # Initialize queue for this group_id if it doesn't exist
        if group_id not in self._episode_queues:
            self._episode_queues[group_id] = asyncio.Queue(maxsize=_MAX_QUEUE_SIZE)

        # Add the episode processing function to the queue
        await self._episode_queues[group_id].put(process_func)

        # Start a worker for this queue if one isn't already running.
        # Store the Task immediately to prevent both GC and duplicate spawns.
        existing = self._queue_workers.get(group_id)
        if existing is None or existing.done():
            task = asyncio.create_task(self._process_episode_queue(group_id))
            self._queue_workers[group_id] = task

        return self._episode_queues[group_id].qsize()

    async def _process_episode_queue(self, group_id: str) -> None:
        """Process episodes for a specific group_id sequentially.

        Runs as a long-lived task. On unexpected errors, restarts with
        exponential backoff (up to 60s) to avoid silent work loss.
        """
        logger.info('Starting episode queue worker for group_id: %s', group_id)
        backoff = 1.0
        max_backoff = 60.0

        while True:
            try:
                while True:
                    process_func = await self._episode_queues[group_id].get()
                    try:
                        await process_func()
                        backoff = 1.0  # reset on success
                    except Exception as e:
                        logger.error(
                            'Error processing queued episode for group_id %s: %s',
                            group_id, type(e).__name__,
                        )
                    finally:
                        self._episode_queues[group_id].task_done()
            except asyncio.CancelledError:
                logger.info('Episode queue worker for group_id %s was cancelled', group_id)
                return  # honour cancellation — do not restart
            except Exception as e:
                remaining = self._episode_queues[group_id].qsize()
                logger.error(
                    'Queue worker for group_id %s crashed (%s), %d items queued. '
                    'Restarting in %.0fs...',
                    group_id, type(e).__name__, remaining, backoff,
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, max_backoff)

    def get_queue_size(self, group_id: str) -> int:
        """Get the current queue size for a group_id."""
        if group_id not in self._episode_queues:
            return 0
        return self._episode_queues[group_id].qsize()

    def is_worker_running(self, group_id: str) -> bool:
        """Check if a worker is running for a group_id."""
        return self._queue_workers.get(group_id, False)

    async def initialize(
        self,
        graphiti_client: Any | None = None,
        client_resolver: Callable[[str], Awaitable[Any]] | None = None,
        ontology_resolver: Callable[[str], tuple[dict | None, str] | dict | None] | None = None,
    ) -> None:
        """Initialize the queue service with client and ontology routing.

        Args:
            graphiti_client: Optional single Graphiti client (legacy fallback)
            client_resolver: Optional async resolver returning a Graphiti client for a group_id
            ontology_resolver: Optional callable returning per-group ontology.
                New-style: returns ``(entity_types, extraction_emphasis)`` tuple.
                Legacy: returns ``entity_types`` dict or None.
                When provided, ``add_episode`` uses it to resolve lane-specific
                entity types and extraction emphasis (passed as
                ``custom_extraction_instructions`` to Graphiti Core).
        """
        if graphiti_client is None and client_resolver is None:
            raise RuntimeError(
                'Queue service initialize() requires graphiti_client or client_resolver.'
            )

        self._graphiti_client = graphiti_client
        self._graphiti_client_resolver = client_resolver
        self._ontology_resolver = ontology_resolver

        if client_resolver is not None:
            logger.info('Queue service initialized with per-group client resolver')
        else:
            logger.info('Queue service initialized with single graphiti client (legacy mode)')

        if ontology_resolver is not None:
            logger.info('Queue service initialized with per-group ontology resolver')

    async def _get_client_for_group(self, group_id: str) -> Any:
        """Resolve the Graphiti client for a group."""
        if self._graphiti_client_resolver is not None:
            return await self._graphiti_client_resolver(group_id)

        if self._graphiti_client is None:
            raise RuntimeError('Queue service not initialized. Call initialize() first.')

        return self._graphiti_client

    async def add_episode(
        self,
        group_id: str,
        name: str,
        content: str,
        source_description: str,
        episode_type: Any,
        entity_types: Any,
        uuid: str | None,
    ) -> int:
        """Add an episode for processing.

        Entity types and extraction emphasis are resolved via the ontology
        resolver when configured. The caller-supplied ``entity_types`` is
        used as a fallback when no resolver is present or it returns None.

        Args:
            group_id: The group ID for the episode
            name: Name of the episode
            content: Episode content
            source_description: Description of the episode source
            episode_type: Type of the episode
            entity_types: Fallback entity types for extraction (used when
                ontology resolver is not configured or has no profile)
            uuid: Episode UUID

        Returns:
            The position in the queue
        """
        if self._graphiti_client is None and self._graphiti_client_resolver is None:
            raise RuntimeError('Queue service not initialized. Call initialize() first.')

        # Resolve lane-specific entity types + extraction emphasis via ontology resolver.
        # Falls back to caller-supplied entity_types (global default).
        resolved_entity_types = entity_types
        resolved_extraction_emphasis: str = ''
        if self._ontology_resolver is not None:
            resolver_result = self._ontology_resolver(group_id)
            # New-style resolver returns (entity_types, extraction_emphasis) tuple.
            # Legacy resolver returns entity_types dict or None.
            if isinstance(resolver_result, tuple):
                per_group, resolved_extraction_emphasis = resolver_result
            else:
                per_group = resolver_result
            if per_group is not None:
                resolved_entity_types = per_group
                logger.info(
                    'Using lane-specific ontology for group %s: %s (emphasis: %d chars)',
                    group_id,
                    list(per_group.keys()),
                    len(resolved_extraction_emphasis),
                )

        async def process_episode():
            """Process the episode using the graphiti client."""
            try:
                logger.info(f'Processing episode {uuid} for group {group_id}')

                client = await self._get_client_for_group(group_id)

                # Process the episode using the graphiti client
                await client.add_episode(
                    name=name,
                    episode_body=content,
                    source_description=source_description,
                    source=episode_type,
                    group_id=group_id,
                    reference_time=datetime.now(timezone.utc),
                    entity_types=resolved_entity_types,
                    uuid=uuid,
                    custom_extraction_instructions=resolved_extraction_emphasis or None,
                )

                logger.info(f'Successfully processed episode {uuid} for group {group_id}')

            except Exception as e:
                logger.error(f'Failed to process episode {uuid} for group {group_id}: {str(e)}')
                raise

        # Use the existing add_episode_task method to queue the processing
        return await self.add_episode_task(group_id, process_episode)
