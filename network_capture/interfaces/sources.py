from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from typing import Optional

from .packet_types import PacketEvent


PacketHandler = Callable[[PacketEvent], None]


class PacketSource(ABC):
    """
    Abstract packet source.

    Implementations:
    - live interface capture (Scapy sniff)
    - PCAP replay (Scapy rdpcap)
    """

    @abstractmethod
    def run(self, handler: PacketHandler, stop_after: Optional[int] = None) -> None:
        raise NotImplementedError

    @abstractmethod
    def iter_events(self, stop_after: Optional[int] = None) -> Iterator[PacketEvent]:
        raise NotImplementedError

