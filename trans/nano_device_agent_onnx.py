#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Jetson Nano device-side ONNX inference node for single-UE HIL evaluation.

Experiment meaning:
  - Jetson Nano is NOT the MEC edge server in this experiment.
  - Jetson Nano is a device-side inference node representing only one target UE
    among the 4 simulated UEs.
  - The PC runs the complete MEC simulator, edge queue, the other virtual UEs,
    reward calculation, delay calculation, and drop calculation.
  - Nano only performs:
        target UE local observation -> ONNX student actor -> target UE action
  - This is single-device hardware-in-the-loop evaluation, not a real
    multi-physical-UE system.
  - Nano acting as a TCP server is only a communication implementation detail;
    it does not mean Nano is the MEC edge server.

Protocol:
  JSON line messages, one JSON object per line.
"""

from __future__ import print_function

import argparse
import json
import socket
import time

import numpy as np


DEFAULT_ONNX_PATH = "mappo_actor_temporal_distilled_p25.onnx"


def import_onnxruntime():
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise ImportError(
            "Could not import onnxruntime: {}. Please install a Jetson-compatible "
            "onnxruntime package before running this script.".format(exc)
        )
    return ort


def build_onnx_session(onnx_path):
    """Create an ONNX Runtime session with CPUExecutionProvider preferred."""
    ort = import_onnxruntime()
    available = ort.get_available_providers()
    if "CPUExecutionProvider" in available:
        providers = ["CPUExecutionProvider"]
    else:
        providers = available
    if not providers:
        raise RuntimeError("No ONNX Runtime execution provider is available.")

    session = ort.InferenceSession(onnx_path, providers=providers)
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if not inputs:
        raise ValueError("ONNX model has no inputs: {}".format(onnx_path))
    if not outputs:
        raise ValueError("ONNX model has no outputs: {}".format(onnx_path))

    input_name = inputs[0].name
    output_names = [output.name for output in outputs]
    return session, input_name, output_names, providers


class JsonLineSocket(object):
    """Small Python 3.6-compatible JSON-line wrapper around a socket."""

    def __init__(self, sock):
        self.sock = sock
        self.buffer = b""

    def recv_json(self):
        while b"\n" not in self.buffer:
            chunk = self.sock.recv(65536)
            if not chunk:
                return None
            self.buffer += chunk
        line, self.buffer = self.buffer.split(b"\n", 1)
        line = line.strip()
        if not line:
            return {}
        return json.loads(line.decode("utf-8"))

    def send_json(self, message):
        data = (json.dumps(message, separators=(",", ":")) + "\n").encode("utf-8")
        self.sock.sendall(data)


def normalize_obs(obs, obs_dim):
    obs_arr = np.asarray(obs, dtype=np.float32)
    if obs_arr.ndim == 2 and obs_arr.shape[0] == 1:
        obs_arr = obs_arr.reshape(-1)
    elif obs_arr.ndim != 1:
        obs_arr = obs_arr.reshape(-1)

    if int(obs_arr.size) != int(obs_dim):
        raise ValueError(
            "Expected one UE observation with obs_dim={}, got shape {} and size {}".format(
                obs_dim, getattr(obs_arr, "shape", None), obs_arr.size
            )
        )
    return obs_arr.reshape(1, int(obs_dim)).astype(np.float32, copy=False)


def action_from_logits(action_output):
    logits = np.asarray(action_output, dtype=np.float32)
    if logits.ndim == 0:
        return int(logits.item())
    if logits.ndim == 1:
        return int(np.argmax(logits, axis=-1).item())
    return int(np.argmax(logits, axis=-1).reshape(-1)[0].item())


def repeat_from_output(repeat_output):
    if repeat_output is None:
        return None
    repeat_arr = np.asarray(repeat_output, dtype=np.float32).reshape(-1)
    if repeat_arr.size == 0:
        return None
    return float(repeat_arr[0])


def run_inference(session, input_name, obs_batch):
    start = time.perf_counter()
    outputs = session.run(None, {input_name: obs_batch})
    infer_ms = (time.perf_counter() - start) * 1000.0

    action = action_from_logits(outputs[0])
    repeat_pred = None
    if len(outputs) >= 2:
        repeat_pred = repeat_from_output(outputs[1])
    return action, infer_ms, repeat_pred


def handle_client(conn, address, session, input_name, obs_dim):
    print("Client connected from {}:{}".format(address[0], address[1]))
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    channel = JsonLineSocket(conn)

    while True:
        msg = channel.recv_json()
        if msg is None:
            print("Client disconnected.")
            return False
        if not isinstance(msg, dict):
            channel.send_json({"type": "error", "error": "message must be a JSON object"})
            continue

        msg_type = msg.get("type")
        if msg_type == "close":
            channel.send_json({"type": "closed"})
            print("Received close message; closing current connection.")
            return False

        if msg_type != "infer":
            channel.send_json(
                {
                    "type": "error",
                    "episode": msg.get("episode"),
                    "step": msg.get("step"),
                    "error": "unsupported message type: {}".format(msg_type),
                }
            )
            continue

        try:
            obs_batch = normalize_obs(msg.get("obs"), obs_dim)
            action, infer_ms, repeat_pred = run_inference(session, input_name, obs_batch)
            response = {
                "type": "action",
                "episode": msg.get("episode"),
                "step": msg.get("step"),
                "action": int(action),
                "infer_ms": float(infer_ms),
            }
            if repeat_pred is not None:
                response["repeat_pred"] = repeat_pred
            channel.send_json(response)
        except Exception as exc:
            channel.send_json(
                {
                    "type": "error",
                    "episode": msg.get("episode"),
                    "step": msg.get("step"),
                    "error": str(exc),
                }
            )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Jetson Nano TCP server for single-target-UE ONNX inference."
    )
    parser.add_argument("--onnx_path", default=DEFAULT_ONNX_PATH)
    parser.add_argument("--obs_dim", type=int, default=10)
    parser.add_argument("--n_actions", type=int, default=2)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    return parser.parse_args()


def main():
    args = parse_args()
    session, input_name, output_names, providers = build_onnx_session(args.onnx_path)

    print("ONNX model path: {}".format(args.onnx_path))
    print("obs_dim: {}".format(args.obs_dim))
    print("n_actions: {}".format(args.n_actions))
    print("ONNX input name: {}".format(input_name))
    print("ONNX output names: {}".format(output_names))
    print("ONNX Runtime providers: {}".format(providers))
    print("Listening on {}:{}".format(args.host, args.port))

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.host, int(args.port)))
    server.listen(1)
    # Poll accept() periodically so Ctrl+C is responsive on Windows terminals.
    server.settimeout(1.0)

    try:
        while True:
            try:
                conn, address = server.accept()
            except socket.timeout:
                continue
            try:
                handle_client(conn, address, session, input_name, args.obs_dim)
            finally:
                conn.close()
    except KeyboardInterrupt:
        print("\nInterrupted; shutting down.")
    finally:
        server.close()


if __name__ == "__main__":
    main()
