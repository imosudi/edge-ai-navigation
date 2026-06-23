"""
app/websocket/stream.py
WebSocket connection manager with multi-channel broadcast support.

Channels:
  camera    - JPEG binary frames
  lidar     - JSON scan data
  fusion    - JSON fused objects
  telemetry - JSON system metrics

Design:
  - Thread-safe using asyncio.Lock per channel
  - Stale/dead connections are silently pruned on send failure
  - Supports future addition of channels without code changes
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import defaultdict
from typing import Any

from fastapi import WebSocket

logger = logging.getLogger(__name__)

# Supported broadcast channels
CHANNELS = ("camera", "lidar", "fusion", "telemetry")


class WebSocketManager:
    """
    Manages WebSocket connections across named channels.

    Usage:
        manager = WebSocketManager()
        await manager.connect(ws, "camera")
        await manager.broadcast_bytes(frame_bytes, "camera")
        await manager.broadcast_json({"key": "val"}, "lidar")
        manager.disconnect(ws, "camera")
    """

    def __init__(self) -> None:
        # channel → list of active WebSocket connections
        self._connections: dict[str, list[WebSocket]] = defaultdict(list)
        # Per-channel lock prevents concurrent mutation during iteration
        self._locks: dict[str, asyncio.Lock] = {ch: asyncio.Lock() for ch in CHANNELS}
        self._broadcast_stats: dict[str, int] = defaultdict(int)

    async def connect(self, websocket: WebSocket, channel: str) -> None:
        """Accept and register a new WebSocket connection on `channel`."""
        await websocket.accept()
        async with self._get_lock(channel):
            self._connections[channel].append(websocket)
        logger.debug(
            "WS connect: channel=%s  total=%d",
            channel,
            len(self._connections[channel]),
        )

    def disconnect(self, websocket: WebSocket, channel: str) -> None:
        """Synchronously remove a WebSocket from the channel's connection list."""
        conns = self._connections[channel]
        if websocket in conns:
            conns.remove(websocket)
        logger.debug(
            "WS disconnect: channel=%s  remaining=%d",
            channel,
            len(conns),
        )

    def connection_count(self, channel: str) -> int:
        """Return number of active connections on `channel`."""
        return len(self._connections.get(channel, []))

    def total_connections(self) -> int:
        """Return total connections across all channels."""
        return sum(len(v) for v in self._connections.values())

    async def broadcast_bytes(self, data: bytes, channel: str) -> None:
        """
        Send raw bytes to all connections on `channel`.

        Used for JPEG video frames to avoid JSON serialisation overhead.
        Dead connections are removed automatically.
        """
        async with self._get_lock(channel):
            live: list[WebSocket] = []
            for ws in self._connections[channel]:
                try:
                    await ws.send_bytes(data)
                    live.append(ws)
                except Exception:
                    # Connection closed or broken - drop silently
                    pass
            self._connections[channel] = live
        self._broadcast_stats[channel] += 1

    async def broadcast_json(self, payload: dict[str, Any], channel: str) -> None:
        """
        Serialise `payload` to JSON and broadcast to all connections on `channel`.

        Falls back to individual removal of dead connections.
        """
        try:
            raw = json.dumps(payload, default=_json_default)
        except (TypeError, ValueError) as exc:
            logger.warning("JSON serialisation error on channel=%s: %s", channel, exc)
            return

        async with self._get_lock(channel):
            live: list[WebSocket] = []
            for ws in self._connections[channel]:
                try:
                    await ws.send_text(raw)
                    live.append(ws)
                except Exception:
                    pass
            self._connections[channel] = live
        self._broadcast_stats[channel] += 1

    def stats(self) -> dict[str, Any]:
        """Return broadcast statistics for monitoring."""
        return {
            "connections": {ch: len(conns) for ch, conns in self._connections.items()},
            "broadcasts":  dict(self._broadcast_stats),
        }

    # ── Internal helpers ────────────────────────────────────────────────────

    def _get_lock(self, channel: str) -> asyncio.Lock:
        if channel not in self._locks:
            self._locks[channel] = asyncio.Lock()
        return self._locks[channel]


def _json_default(obj: Any) -> Any:
    """Custom JSON serialiser for numpy/non-standard types."""
    try:
        import numpy as np
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:
        pass
    return str(obj)
