from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


L4Proto = Literal["TCP", "UDP", "OTHER"]


@dataclass(frozen=True)
class PacketEvent:
    """
    Minimal packet metadata event for the real-time pipeline.

    Note: This is intentionally small and stable. Later phases can add:
    - flow keys, TCP flags, payload stats, TLS/HTTP/DNS parsed fields, etc.
    """

    timestamp: float
    src_ip: str
    dst_ip: str
    src_port: Optional[int]
    dst_port: Optional[int]
    l4_proto: L4Proto
    ip_proto: Optional[int]
    length: int
    is_dns: bool = False
    summary: str = ""

