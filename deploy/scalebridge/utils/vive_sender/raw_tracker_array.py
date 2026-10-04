import time
from typing import Dict, List, Optional, Sequence, Tuple


RAW_TRACKER_ROW_SIZE = 14


def _ordered_device_names(devices: Dict[str, dict], requested_names: Optional[Sequence[str]] = None) -> List[str]:
    if requested_names:
        return [device_name for device_name in requested_names if device_name in devices]
    return list(devices.keys())


def _vector_components(vector: Optional[dict], keys: Sequence[str]) -> List[float]:
    if vector is None:
        return [0.0 for _ in keys]
    return [float(vector.get(key, 0.0)) for key in keys]


def pack_raw_tracker_array(devices: Dict[str, dict], device_names: Optional[Sequence[str]] = None, timestamp: Optional[float] = None) -> List[List[float]]:
    sample_timestamp = float(time.time() if timestamp is None else timestamp)
    rows = []
    for device_name in _ordered_device_names(devices, device_names):
        device_data = devices[device_name]
        row = []
        row.extend(_vector_components(device_data.get("position"), ("x", "y", "z")))
        row.extend(_vector_components(device_data.get("quaternion"), ("w", "x", "y", "z")))
        row.extend(_vector_components(device_data.get("velocity"), ("x", "y", "z")))
        row.extend(_vector_components(device_data.get("angular_velocity"), ("x", "y", "z")))
        row.append(sample_timestamp)
        rows.append(row)
        print(f"Packed device '{device_name}' into row: {row}")
    return rows


def read_raw_tracker_array(reader, device_names: Optional[Sequence[str]] = None) -> Tuple[List[str], List[List[float]], float]:
    reader.poll_events()
    timestamp = time.time()
    devices = reader.read_devices(device_names=device_names)
    ordered_names = _ordered_device_names(devices, device_names)
    return ordered_names, pack_raw_tracker_array(devices, device_names=ordered_names, timestamp=timestamp), timestamp