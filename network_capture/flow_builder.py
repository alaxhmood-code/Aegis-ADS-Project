"""
Flow builder (stub).

Next phase (not implemented here):
- Convert PacketEvent stream into flow records (5-tuple key, time windows)
- Track TCP state, bytes/packets counters, IAT, flags, etc.
- Emit FlowEvent objects into the next queue stage
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class FlowKey:
    src_ip: str
    dst_ip: str
    src_port: Optional[int]
    dst_port: Optional[int]
    l4_proto: str


class FlowBuilder:
    def __init__(self) -> None:
        # Intentionally empty: real implementation comes later.
        pass

