"""Remote VGGT-Ω client; array bytes follow JSON without a frame prefix."""

import json
import socket
from pathlib import Path

import numpy as np

from perception.wire import receive_arrays, send_frame


class RemoteVGGT:
    def __init__(self, host: str, port: int, timeout: float = 900):
        self.host, self.port, self.timeout = host, port, timeout
        self.metadata = {"backend": "remote", "model": "vggt", "host": host, "port": port}

    def infer(self, frames: list[Path]) -> dict[str, np.ndarray]:
        blobs = [path.read_bytes() for path in frames]
        if len(blobs) != 32:
            raise ValueError("VGGT requires 32 ordered frames")
        header = {"type": "infer", "model": "vggt", "mode": "images",
                  "images_bytes": [len(blob) for blob in blobs],
                  "params": {"resolution": 512, "preprocess": "max_size", "num_frames": 32}}
        with socket.create_connection((self.host, self.port), self.timeout) as connection:
            connection.settimeout(self.timeout)
            send_frame(connection, json.dumps(header).encode())
            send_frame(connection, b"".join(blobs))
            arrays, response = receive_arrays(connection, framed=False)
            self.metadata = {**response, "backend": "remote", "model": "vggt",
                             "host": self.host, "port": self.port}
        required = {"depth": (32, 288, 512), "depth_conf": (32, 288, 512),
                    "intrinsics": (32, 3, 3), "extrinsics": (32, 3, 4)}
        for name, shape in required.items():
            if name not in arrays or arrays[name].shape != shape or not np.isfinite(arrays[name]).all():
                raise ValueError(f"VGGT returned invalid {name}; expected {shape}")
        return arrays
