import time
from typing import Dict, Iterator, List, Optional

import tracker


class ViveTrackerReader:
    def __init__(
        self,
        configfile_path: Optional[str] = None,
        vr_system=None,
        position_epsilon: float = 1e-4,
        quaternion_epsilon: float = 1e-4,
    ):
        self.vr_system = vr_system or tracker.triad_openvr(configfile_path=configfile_path)
        self.frame_index = 0
        self.position_epsilon = position_epsilon
        self.quaternion_epsilon = quaternion_epsilon
        self.previous_devices: Optional[Dict[str, dict]] = None
        self.previous_sample_monotonic_ns: Optional[int] = None
        self.duplicate_streak = 0

    def list_devices(self) -> List[str]:
        return self.vr_system.get_device_names()

    def print_discovered_objects(self) -> None:
        self.vr_system.print_discovered_objects()

    def read_devices(self, device_names=None):
        return self.vr_system.get_devices_data(device_names=device_names)

    def poll_events(self) -> None:
        poll_vr_events = getattr(self.vr_system, "poll_vr_events", None)
        if callable(poll_vr_events):
            poll_vr_events()

    def _resolve_device_order(self, requested_names, devices: Dict[str, dict]) -> List[str]:
        if requested_names:
            return [device_name for device_name in requested_names if device_name in devices]
        return list(devices.keys())

    def _copy_vector(self, vector):
        if vector is None:
            return None
        return dict(vector)

    def _snapshot_devices(self, devices: Dict[str, dict]) -> Dict[str, dict]:
        snapshot = {}
        for device_name, device_data in devices.items():
            snapshot[device_name] = {
                "position": self._copy_vector(device_data.get("position")),
                "quaternion": self._copy_vector(device_data.get("quaternion")),
                "velocity": self._copy_vector(device_data.get("velocity")),
                "angular_velocity": self._copy_vector(device_data.get("angular_velocity")),
            }
        return snapshot

    def _max_vector_delta(self, current_vector, previous_vector) -> float:
        if current_vector is None and previous_vector is None:
            return 0.0
        if current_vector is None or previous_vector is None:
            return float("inf")

        all_keys = set(current_vector.keys()) | set(previous_vector.keys())
        return max(abs(float(current_vector.get(key, 0.0)) - float(previous_vector.get(key, 0.0))) for key in all_keys)

    def _build_stability(self, devices: Dict[str, dict], sample_monotonic_ns: int) -> dict:
        sample_interval_ms = None
        if self.previous_sample_monotonic_ns is not None:
            sample_interval_ms = (sample_monotonic_ns - self.previous_sample_monotonic_ns) / 1_000_000.0

        changed_device_names = []
        new_device_names = []
        missing_device_names = []
        device_deltas = {}
        max_position_delta = 0.0
        max_quaternion_delta = 0.0

        if self.previous_devices is not None:
            current_names = set(devices.keys())
            previous_names = set(self.previous_devices.keys())
            new_device_names = sorted(current_names - previous_names)
            missing_device_names = sorted(previous_names - current_names)

            for device_name in sorted(current_names & previous_names):
                current_device = devices[device_name]
                previous_device = self.previous_devices[device_name]
                position_delta = self._max_vector_delta(current_device.get("position"), previous_device.get("position"))
                quaternion_delta = self._max_vector_delta(current_device.get("quaternion"), previous_device.get("quaternion"))
                max_position_delta = max(max_position_delta, position_delta)
                max_quaternion_delta = max(max_quaternion_delta, quaternion_delta)

                if position_delta > self.position_epsilon or quaternion_delta > self.quaternion_epsilon:
                    changed_device_names.append(device_name)
                    device_deltas[device_name] = {
                        "position_max_delta": position_delta,
                        "quaternion_max_delta": quaternion_delta,
                    }

        is_duplicate_frame = (
            self.previous_devices is not None
            and not changed_device_names
            and not new_device_names
            and not missing_device_names
        )

        if is_duplicate_frame:
            self.duplicate_streak += 1
        else:
            self.duplicate_streak = 0

        stability = {
            "is_duplicate_frame": is_duplicate_frame,
            "duplicate_streak": self.duplicate_streak,
            "sample_interval_ms": sample_interval_ms,
            "changed_device_names": changed_device_names,
            "new_device_names": new_device_names,
            "missing_device_names": missing_device_names,
            "max_position_delta": max_position_delta,
            "max_quaternion_delta": max_quaternion_delta,
            "device_deltas": device_deltas,
            "position_epsilon": self.position_epsilon,
            "quaternion_epsilon": self.quaternion_epsilon,
        }

        self.previous_devices = self._snapshot_devices(devices)
        self.previous_sample_monotonic_ns = sample_monotonic_ns
        return stability

    def build_frame(self, device_names=None):
        self.poll_events()
        sample_monotonic_ns = time.perf_counter_ns()
        timestamp_ns = time.time_ns()
        devices = self.read_devices(device_names=device_names)
        stability = self._build_stability(devices, sample_monotonic_ns)
        frame = {
            "frame_index": self.frame_index,
            "timestamp": timestamp_ns / 1_000_000_000.0,
            "timestamp_ns": timestamp_ns,
            "device_count": len(devices),
            "device_order": self._resolve_device_order(device_names, devices),
            "coordinate_system": "raw_openvr",
            "cursor": {
                "sequence": self.frame_index,
                "monotonic_ns": sample_monotonic_ns,
                "wall_time_ns": timestamp_ns,
                "suppressed_duplicates_since_last_emit": 0,
            },
            "stability": stability,
            "devices": devices,
        }
        self.frame_index += 1
        return frame

    def iter_frames(self, update_hz: float = 250.0, device_names=None, emit_unchanged: bool = True) -> Iterator[dict]:
        interval_ns = int(1_000_000_000 / update_hz) if update_hz > 0 else 0
        next_deadline_ns = time.perf_counter_ns()
        suppressed_duplicates = 0

        while True:
            if interval_ns > 0:
                now_ns = time.perf_counter_ns()
                remaining_ns = next_deadline_ns - now_ns
                if remaining_ns > 0:
                    if remaining_ns > 500_000:
                        time.sleep((remaining_ns - 200_000) / 1_000_000_000.0)
                    while time.perf_counter_ns() < next_deadline_ns:
                        pass

            frame = self.build_frame(device_names=device_names)
            if emit_unchanged or not frame["stability"]["is_duplicate_frame"]:
                frame["cursor"]["suppressed_duplicates_since_last_emit"] = suppressed_duplicates
                suppressed_duplicates = 0
                yield frame
            else:
                suppressed_duplicates += 1

            if interval_ns <= 0:
                continue

            next_deadline_ns += interval_ns
            now_ns = time.perf_counter_ns()
            if now_ns > next_deadline_ns:
                missed_intervals = (now_ns - next_deadline_ns) // interval_ns
                if missed_intervals > 0:
                    next_deadline_ns += missed_intervals * interval_ns


class RelativeZTrackerReader(ViveTrackerReader):
    def read_devices(self, device_names=None):
        return self.vr_system.get_transformed_devices_data(
            device_names=device_names,
            relative_to_first_device_z=True,
        )

    def build_frame(self, device_names=None):
        frame = super().build_frame(device_names=device_names)
        frame["coordinate_system"] = "x_forward_y_left_z_up"
        frame["z_reference_device"] = frame["device_order"][0] if frame["device_order"] else None
        frame["z_reference_rule"] = "all device z values are relative to the first valid device"
        return frame
