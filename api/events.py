# pushing what just happened to whoever is watching
#
# the alert store is the record; this is the doorbell. when a live scan lands
# or an alert is published, every open page should hear about it without
# asking. server-sent events do that with no dependency and no protocol
# upgrade - it is an ordinary HTTP response that never ends, which means it
# survives proxies and static hosts that would block a websocket.
#
# publishing happens on a worker thread and reading happens on the event loop,
# so the queue between them is a plain thread-safe queue.Queue and the reader
# polls it. a subscriber that stops reading has its queue fill and is dropped
# rather than being allowed to grow without limit.

from __future__ import annotations

import json
import queue
import threading
from collections import deque

import pandas as pd

MAX_QUEUE = 64          # per subscriber, then we drop them
HISTORY = 50            # recent events replayed to a page that just opened

_lock = threading.Lock()
_subscribers: list[queue.Queue] = []
_history: deque = deque(maxlen=HISTORY)
_seq = 0


def publish(kind: str, payload: dict) -> dict:
    # kind is what the page switches on: "alert", "fetch", "scan", "heartbeat"
    global _seq
    with _lock:
        _seq += 1
        event = {"seq": _seq, "kind": kind,
                 "at": pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                 **payload}
        if kind != "heartbeat":
            _history.append(event)
        dead = []
        for q in _subscribers:
            try:
                q.put_nowait(event)
            except queue.Full:
                dead.append(q)
        for q in dead:
            _subscribers.remove(q)
    return event


def history() -> list[dict]:
    with _lock:
        return list(_history)


def subscribe() -> queue.Queue:
    q: queue.Queue = queue.Queue(maxsize=MAX_QUEUE)
    with _lock:
        _subscribers.append(q)
    return q


def unsubscribe(q: queue.Queue) -> None:
    with _lock:
        if q in _subscribers:
            _subscribers.remove(q)


def subscriber_count() -> int:
    with _lock:
        return len(_subscribers)


def sse(event: dict) -> str:
    # one server-sent event. the blank line terminates it; without it the
    # browser buffers forever and the page looks dead.
    return f"event: {event['kind']}\ndata: {json.dumps(event)}\n\n"
