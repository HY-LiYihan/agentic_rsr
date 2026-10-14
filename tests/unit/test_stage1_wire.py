import json
import socket
import struct
import threading
import time

import numpy as np
import pytest

from perception.sam3.remote_client import RemoteSAM3
from perception.vggt_omega.remote_client import RemoteVGGT
from perception.wire import read_frame, send_frame


def server(handler):
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]

    def serve():
        with listener:
            connection, _ = listener.accept()
            with connection:
                handler(connection)

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    return port, thread


def chunked(connection, data):
    for start in range(0, len(data), 3000):
        connection.sendall(data[start:start + 3000])


def respond(connection, name, array, framed):
    message = {"type": "array", "name": name, "shape": array.shape,
               "dtype": str(array.dtype), "bytes": array.nbytes}
    send_frame(connection, json.dumps(message).encode())
    payload = array.tobytes()
    if framed:
        payload = struct.pack("<I", len(payload)) + payload
    chunked(connection, payload)


def test_vggt_unframed_array_and_request(tmp_path):
    paths = [tmp_path / f"{index}.jpg" for index in range(32)]
    for path in paths:
        path.write_bytes(b"jpg")

    def handler(connection):
        header = json.loads(read_frame(connection))
        assert header["model"] == "vggt" and header["images_bytes"] == [3] * 32
        assert read_frame(connection) == b"jpg" * 32
        send_frame(connection, b'{"type":"status","stage":"inferring"}')
        respond(connection, "depth", np.ones((32, 288, 512), np.float32), False)
        respond(connection, "depth_conf", np.ones((32, 288, 512), np.float32), False)
        respond(connection, "intrinsics", np.tile(np.eye(3, dtype=np.float32), (32, 1, 1)), False)
        respond(connection, "extrinsics", np.zeros((32, 3, 4), np.float32), False)
        send_frame(connection, b'{"type":"done","frames":32,"backend":"local"}')

    port, thread = server(handler)
    client = RemoteVGGT("127.0.0.1", port, timeout=10)
    assert client.infer(paths)["depth"].shape == (32, 288, 512)
    assert client.metadata["frames"] == 32
    assert client.metadata["backend"] == "remote"
    thread.join(timeout=2)


def test_sam_framed_arrays_and_empty(tmp_path):
    image = tmp_path / "image.jpg"
    image.write_bytes(b"data")

    def handler(connection):
        header = json.loads(read_frame(connection))
        assert header["model"] == "sam3" and header["text"] == ["robotic arm with base and camera"]
        assert read_frame(connection) == b"data"
        respond(connection, "masks", np.zeros((0, 288, 512), np.uint8), True)
        send_frame(connection, b'{"type":"done","k":0}')

    port, thread = server(handler)
    assert not RemoteSAM3("127.0.0.1", port).segment(image).any()
    thread.join(timeout=2)


def test_remote_error_and_timeout(tmp_path):
    image = tmp_path / "image.jpg"
    image.write_bytes(b"image")

    def failure(connection):
        read_frame(connection)
        read_frame(connection)
        send_frame(connection, b'{"type":"error","message":"bad request"}')

    port, thread = server(failure)
    with pytest.raises(RuntimeError, match="bad request"):
        RemoteSAM3("127.0.0.1", port).segment(image)
    thread.join(timeout=2)

    def slow(connection):
        read_frame(connection)
        read_frame(connection)
        time.sleep(0.3)

    port, thread = server(slow)
    with pytest.raises((TimeoutError, socket.timeout)):
        RemoteSAM3("127.0.0.1", port, timeout=0.05).segment(image)
    thread.join(timeout=2)
