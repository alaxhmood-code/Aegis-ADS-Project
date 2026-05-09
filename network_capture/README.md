# Network Capture Layer (Real-Time IDS Foundation)

This directory adds a **traffic collection layer** for Aegis-ADS.

Goal (current phase):

`Network Traffic → Packet Capture Layer → (Queue/Buffer)`

Not included yet:
- flow reconstruction / feature extraction
- AI inference / scoring
- dashboards, APIs, databases

## Quick start (live capture)

Prereqs:
- Python package: `scapy`
- OS packet capture support (on Windows this typically requires Npcap)
- Run with sufficient privileges to capture packets.

Example:
```powershell
python -m network_capture.packet_sniffer --iface "Ethernet" --bpf "tcp or udp or port 53"
```

## Quick start (PCAP read)
```powershell
python -m network_capture.packet_sniffer --pcap .\path\to\capture.pcap
```

## Structure
- `packet_sniffer.py`: capture packets (live or PCAP), parse basics, push into a queue
- `interfaces/`: clean interfaces/typing for future sensors/sources
- `utils/`: logging + helpers
- `flow_builder.py`: stub (next phase)
- `feature_extractor.py`: stub (next phase)

