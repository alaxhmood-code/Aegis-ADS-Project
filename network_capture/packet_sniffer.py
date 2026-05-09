from __future__ import annotations

import argparse
import logging
import queue
import time
from dataclasses import replace
from typing import Iterator, Optional

from network_capture.interfaces.packet_types import PacketEvent
from network_capture.interfaces.sources import PacketSource
from network_capture.utils.logging_config import configure_logging
from network_capture.utils.packet_parse import build_summary, safe_int


class PacketBuffer:
    """
    Minimal queue/buffer between capture and downstream processing.

    Drop policy: if full, drop newest event and log a warning (keeps capture thread moving).
    """

    def __init__(self, maxsize: int = 10_000, logger: Optional[logging.Logger] = None) -> None:
        self._q: queue.Queue[PacketEvent] = queue.Queue(maxsize=maxsize)
        self._logger = logger or logging.getLogger("aegis.capture")
        self.dropped = 0

    def put(self, evt: PacketEvent) -> None:
        try:
            self._q.put_nowait(evt)
        except queue.Full:
            self.dropped += 1
            if self.dropped % 100 == 1:
                self._logger.warning("packet_buffer_full dropped=%d", self.dropped)

    def get(self, timeout: Optional[float] = None) -> PacketEvent:
        return self._q.get(timeout=timeout)

    def qsize(self) -> int:
        return self._q.qsize()


class ScapyLiveSource(PacketSource):
    def __init__(
        self,
        iface: Optional[str],
        bpf: Optional[str],
        logger: logging.Logger,
    ) -> None:
        self._iface = iface
        self._bpf = bpf
        self._logger = logger

    def iter_events(self, stop_after: Optional[int] = None) -> Iterator[PacketEvent]:
        # Scapy import is intentionally local to keep the module importable without scapy installed.
        try:
            from scapy.all import IP, TCP, UDP, DNS, sniff  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "Scapy is required for live capture. Install scapy and ensure capture drivers exist."
            ) from e

        count = 0

        def to_event(pkt) -> Optional[PacketEvent]:
            if not pkt.haslayer(IP):
                return None
            ip = pkt[IP]
            src_ip = getattr(ip, "src", "0.0.0.0")
            dst_ip = getattr(ip, "dst", "0.0.0.0")
            ip_proto = safe_int(getattr(ip, "proto", None))

            l4_proto = "OTHER"
            src_port = None
            dst_port = None
            is_dns = False

            if pkt.haslayer(TCP):
                l4_proto = "TCP"
                tcp = pkt[TCP]
                src_port = safe_int(getattr(tcp, "sport", None))
                dst_port = safe_int(getattr(tcp, "dport", None))
            elif pkt.haslayer(UDP):
                l4_proto = "UDP"
                udp = pkt[UDP]
                src_port = safe_int(getattr(udp, "sport", None))
                dst_port = safe_int(getattr(udp, "dport", None))

            if pkt.haslayer(DNS):
                is_dns = True

            ts = float(getattr(pkt, "time", time.time()))
            length = len(bytes(pkt))
            evt = PacketEvent(
                timestamp=ts,
                src_ip=str(src_ip),
                dst_ip=str(dst_ip),
                src_port=src_port,
                dst_port=dst_port,
                l4_proto=l4_proto,  # type: ignore[arg-type]
                ip_proto=ip_proto,
                length=length,
                is_dns=is_dns,
            )
            return replace(evt, summary=build_summary(evt))

        def on_packet(pkt) -> None:
            nonlocal count
            evt = to_event(pkt)
            if evt is None:
                return
            count += 1
            yield_queue.put(evt)
            if count % 50 == 0:
                self._logger.info("captured_packets count=%d queue=%d dropped=%d", count, yield_queue.qsize(), yield_queue.dropped)
            self._logger.debug("packet %s", evt.summary)

        yield_queue = PacketBuffer(maxsize=10_000, logger=self._logger)

        # Use sniff in a blocking fashion, but we also return an iterator by draining the queue.
        # Start sniffing in the same thread; consumers can iterate concurrently only if caller manages threading.
        self._logger.info("live_capture_start iface=%s bpf=%s", self._iface, self._bpf)
        sniff(
            iface=self._iface,
            filter=self._bpf,
            prn=on_packet,
            store=False,
            count=stop_after or 0,
        )
        self._logger.info("live_capture_stop count=%d dropped=%d", count, yield_queue.dropped)

        # Drain any remaining events captured at the end.
        while True:
            try:
                yield yield_queue.get(timeout=0.01)
            except queue.Empty:
                break

    def run(self, handler, stop_after: Optional[int] = None) -> None:
        for evt in self.iter_events(stop_after=stop_after):
            handler(evt)


class ScapyPcapSource(PacketSource):
    def __init__(self, pcap_path: str, logger: logging.Logger) -> None:
        self._pcap_path = pcap_path
        self._logger = logger

    def iter_events(self, stop_after: Optional[int] = None) -> Iterator[PacketEvent]:
        try:
            from scapy.all import IP, TCP, UDP, DNS, PcapReader  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("Scapy is required for PCAP reading. Install scapy.") from e

        self._logger.info("pcap_read_start path=%s", self._pcap_path)
        count = 0
        with PcapReader(self._pcap_path) as reader:
            for pkt in reader:
                if stop_after is not None and count >= stop_after:
                    break
                if not pkt.haslayer(IP):
                    continue
                ip = pkt[IP]
                src_ip = getattr(ip, "src", "0.0.0.0")
                dst_ip = getattr(ip, "dst", "0.0.0.0")
                ip_proto = safe_int(getattr(ip, "proto", None))

                l4_proto = "OTHER"
                src_port = None
                dst_port = None
                is_dns = False

                if pkt.haslayer(TCP):
                    l4_proto = "TCP"
                    tcp = pkt[TCP]
                    src_port = safe_int(getattr(tcp, "sport", None))
                    dst_port = safe_int(getattr(tcp, "dport", None))
                elif pkt.haslayer(UDP):
                    l4_proto = "UDP"
                    udp = pkt[UDP]
                    src_port = safe_int(getattr(udp, "sport", None))
                    dst_port = safe_int(getattr(udp, "dport", None))

                if pkt.haslayer(DNS):
                    is_dns = True

                ts = float(getattr(pkt, "time", time.time()))
                length = len(bytes(pkt))
                evt = PacketEvent(
                    timestamp=ts,
                    src_ip=str(src_ip),
                    dst_ip=str(dst_ip),
                    src_port=src_port,
                    dst_port=dst_port,
                    l4_proto=l4_proto,  # type: ignore[arg-type]
                    ip_proto=ip_proto,
                    length=length,
                    is_dns=is_dns,
                )
                evt = replace(evt, summary=build_summary(evt))
                count += 1
                yield evt

        self._logger.info("pcap_read_stop count=%d", count)

    def run(self, handler, stop_after: Optional[int] = None) -> None:
        for evt in self.iter_events(stop_after=stop_after):
            handler(evt)


def _build_source(args, logger: logging.Logger) -> PacketSource:
    if args.pcap:
        return ScapyPcapSource(args.pcap, logger)
    return ScapyLiveSource(args.iface, args.bpf, logger)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Aegis-ADS packet capture (foundation layer).")
    parser.add_argument("--iface", help="Interface name for live capture (e.g., Ethernet, eth0).")
    parser.add_argument("--pcap", help="Read packets from a PCAP file instead of live capture.")
    parser.add_argument(
        "--bpf",
        default="tcp or udp or port 53",
        help="BPF filter for live capture (default: tcp or udp or port 53).",
    )
    parser.add_argument("--count", type=int, default=0, help="Stop after N packets (0 = unlimited).")
    parser.add_argument("--log-file", default=None, help="Optional log file path.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging (per-packet summaries).")
    args = parser.parse_args(argv)

    level = logging.DEBUG if args.debug else logging.INFO
    logger = configure_logging(level=level, log_file=args.log_file)

    if not args.pcap and not args.iface:
        logger.error("missing_capture_source specify --iface for live or --pcap for file replay")
        return 2

    source = _build_source(args, logger)

    buffer = PacketBuffer(maxsize=10_000, logger=logger)

    def handler(evt: PacketEvent) -> None:
        buffer.put(evt)
        # Live display (human operator): summary only.
        logger.info("packet %s", evt.summary)

    stop_after = args.count if args.count and args.count > 0 else None
    try:
        source.run(handler, stop_after=stop_after)
    except Exception as e:
        logger.exception("capture_failed error=%s", e)
        return 1

    logger.info("capture_done buffered=%d dropped=%d", buffer.qsize(), buffer.dropped)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

