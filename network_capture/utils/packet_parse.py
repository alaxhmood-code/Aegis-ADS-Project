from __future__ import annotations

from typing import Optional

from network_capture.interfaces.packet_types import PacketEvent


def safe_int(value: object) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def build_summary(evt: PacketEvent) -> str:
    l4 = evt.l4_proto
    sp = evt.src_port if evt.src_port is not None else "-"
    dp = evt.dst_port if evt.dst_port is not None else "-"
    dns = " DNS" if evt.is_dns else ""
    return f"{l4}{dns} {evt.src_ip}:{sp} -> {evt.dst_ip}:{dp} len={evt.length}"

