import argparse
import json
import sys
import time

from raw_tracker_array import RAW_TRACKER_ROW_SIZE, read_raw_tracker_array
from tcp_transport import TcpJsonSender
from tracker_reader import ViveTrackerReader


def parse_args():
    parser = argparse.ArgumentParser(description="Send raw Vive tracker samples as an (n, 14) JSON array over TCP.")
    parser.add_argument("host", help="Target receiver IP or hostname")
    parser.add_argument("devices", nargs="*", help="Optional device names to stream")
    parser.add_argument("--port", type=int, default=5000, help="Target TCP port")
    parser.add_argument("--hz", type=float, default=250.0, help="Streaming rate in Hz")
    parser.add_argument("--status-every", type=int, default=25, help="Print one status line every N sent arrays")
    return parser.parse_args()


def validate_requested_devices(reader, requested_names):
    available_devices = reader.list_devices()
    if not requested_names:
        return available_devices

    missing_names = [name for name in requested_names if name not in available_devices]
    if missing_names:
        print("Unknown device(s): " + ", ".join(missing_names))
        sys.exit(1)

    return requested_names


def main():
    args = parse_args()
    reader = ViveTrackerReader()
    reader.print_discovered_objects()
    selected_devices = validate_requested_devices(reader, args.devices)
    if not selected_devices:
        print("No tracked devices found.")
        sys.exit(1)

    sent_arrays = 0
    with TcpJsonSender(args.host, args.port) as sender:
        interval_ns = int(1_000_000_000 / args.hz) if args.hz > 0 else 0
        next_deadline_ns = time.perf_counter_ns()

        while True:
            if interval_ns > 0:
                now_ns = time.perf_counter_ns()
                remaining_ns = next_deadline_ns - now_ns
                if remaining_ns > 0:
                    if remaining_ns > 500_000:
                        time.sleep((remaining_ns - 200_000) / 1_000_000_000.0)
                    while time.perf_counter_ns() < next_deadline_ns:
                        pass

            device_names, payload, timestamp = read_raw_tracker_array(reader, device_names=selected_devices)
            sender.send_message(payload)
            sent_arrays += 1

            if sent_arrays == 1 or sent_arrays % max(args.status_every, 1) == 0:
                print(
                    "\r"
                    f"arrays={sent_arrays} shape=({len(payload)}, {RAW_TRACKER_ROW_SIZE}) devices={','.join(device_names)} "
                    f"target={args.host}:{args.port} timestamp={timestamp:.6f}",
                    flush=True,
                )
                print(json.dumps(payload, ensure_ascii=False))

            if interval_ns <= 0:
                continue

            next_deadline_ns += interval_ns
            now_ns = time.perf_counter_ns()
            if now_ns > next_deadline_ns:
                missed_intervals = (now_ns - next_deadline_ns) // interval_ns
                if missed_intervals > 0:
                    next_deadline_ns += missed_intervals * interval_ns


if __name__ == "__main__":
    main()