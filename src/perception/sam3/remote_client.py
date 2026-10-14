"""Remote SAM3 text segmentation; arrays arrive as separate framed payloads."""

import json
import socket
from pathlib import Path

import numpy as np

from perception.wire import receive_arrays, send_frame
from perception.sam3 import DEFAULT_PROMPT
from perception.sam3.instances import validate_instances


class RemoteSAM3:
    def __init__(self, host: str, port: int, timeout: float = 180):
        self.host, self.port, self.timeout = host, port, timeout
        self.metadata = {"backend": "remote", "model": "sam3", "host": host, "port": port}

    def _request(self, image: Path, prompt: str) -> dict:
        blob = image.read_bytes()
        header = {"type": "infer", "model": "sam3", "mode": "text",
                  "image_bytes": len(blob), "text": [prompt], "conf": 0.25}
        with socket.create_connection((self.host, self.port), self.timeout) as connection:
            connection.settimeout(self.timeout)
            send_frame(connection, json.dumps(header).encode())
            send_frame(connection, blob)
            arrays, response = receive_arrays(connection, framed=True)
            self.metadata = {**response, "backend": "remote", "model": "sam3",
                             "host": self.host, "port": self.port}
        return arrays

    def segment(self, image: Path, prompt: str = DEFAULT_PROMPT) -> np.ndarray:
        arrays = self._request(image, prompt)
        masks = arrays.get("masks")
        if masks is None or masks.ndim != 3 or masks.shape[1:] != (288, 512):
            raise ValueError("SAM3 returned an invalid mask resolution")
        return np.any(masks > 0, axis=0)

    def segment_instances(self, image: Path, prompt: str = DEFAULT_PROMPT,
                          expected_shape: tuple[int, int] | None = None) -> dict:
        if expected_shape is None:
            from PIL import Image

            with Image.open(image) as source:
                expected_shape = (source.height, source.width)
        arrays = self._request(image, prompt)
        masks, scores = arrays.get("masks"), arrays.get("scores")
        if masks is not None and np.ndim(masks) == 3 and len(masks) == 0 and scores is None:
            scores = np.zeros(0, dtype=np.float32)
        return validate_instances(masks, scores, expected_shape)

    def close(self):
        pass
