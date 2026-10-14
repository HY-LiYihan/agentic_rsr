"""Length-prefixed inference service framing shared by both remote models."""

import json
import math
import socket
import struct

import numpy as np


MAX_BYTES = 512 * 1024 * 1024


def read_exact(connection: socket.socket, size: int) -> bytes:
    if size < 0 or size > MAX_BYTES:
        raise ValueError(f"invalid payload size: {size}")
    chunks = bytearray()
    while len(chunks) < size:
        chunk = connection.recv(min(size - len(chunks), 1024 * 1024))
        if not chunk:
            raise ConnectionError("inference service closed the connection")
        chunks.extend(chunk)
    return bytes(chunks)


def send_frame(connection: socket.socket, payload: bytes) -> None:
    if len(payload) > MAX_BYTES:
        raise ValueError("inference request exceeds maximum payload size")
    connection.sendall(struct.pack("<I", len(payload)) + payload)


def read_frame(connection: socket.socket) -> bytes:
    size = struct.unpack("<I", read_exact(connection, 4))[0]
    return read_exact(connection, size)


def read_message(connection: socket.socket) -> dict:
    message = json.loads(read_frame(connection))
    if not isinstance(message, dict) or "type" not in message:
        raise ValueError("malformed inference message")
    return message


def read_array(connection: socket.socket, message: dict, *, framed: bool) -> np.ndarray:
    dtype = np.dtype(message["dtype"])
    if dtype.hasobject or dtype.itemsize == 0 or dtype.byteorder == ">":
        raise ValueError("unsupported array dtype")
    shape = tuple(int(dimension) for dimension in message["shape"])
    if not shape or any(dimension < 0 for dimension in shape):
        raise ValueError("invalid array shape")
    size = math.prod(shape) * dtype.itemsize
    if size > MAX_BYTES:
        raise ValueError("inference array exceeds maximum size")
    if "bytes" in message and message["bytes"] != size:
        raise ValueError("inference array header size mismatch")
    payload = read_frame(connection) if framed else read_exact(connection, size)
    if len(payload) != size:
        raise ValueError("inference array payload size mismatch")
    return np.frombuffer(payload, dtype=dtype).reshape(shape).copy()


def receive_arrays(connection: socket.socket, *, framed: bool) -> tuple[dict[str, np.ndarray], dict]:
    arrays: dict[str, np.ndarray] = {}
    while True:
        message = read_message(connection)
        kind = message["type"]
        if kind == "array":
            name = message["name"]
            if name in arrays:
                raise ValueError(f"duplicate inference array: {name}")
            arrays[name] = read_array(connection, message, framed=framed)
        elif kind == "done":
            return arrays, message
        elif kind == "error":
            raise RuntimeError(message.get("message", "inference service error"))
        elif kind != "status":
            raise ValueError(f"unexpected inference message: {kind}")
