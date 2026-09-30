"""人工确认（human-in-the-loop）—— 块 C 的确认点，接到流式事件上的实现。"""
from __future__ import annotations

import threading
import time
import uuid

DEFAULT_TIMEOUT_S = 120.0


class ConfirmationRegistry:
    """待确认项的登记表。"""

    def __init__(self, timeout_s: float = DEFAULT_TIMEOUT_S) -> None:
        self.timeout_s = timeout_s
        self._pending: dict[str, dict] = {}
        self._lock = threading.Lock()

    def open(self, *, tool: str, args: str, request_id: str) -> str:
        """登记一个待确认项，返回 confirm_id。"""
        cid = uuid.uuid4().hex[:12]
        with self._lock:
            self._pending[cid] = {
                "confirm_id": cid,
                "tool": tool,
                "args": args,
                "request_id": request_id,
                "event": threading.Event(),
                "approved": None,
                "created_at": time.time(),
            }
        return cid

    def wait(self, confirm_id: str, timeout_s: float | None = None) -> bool:
        """阻塞等决定。超时或未登记 → False（拒绝）。"""
        with self._lock:
            rec = self._pending.get(confirm_id)
        if rec is None:
            return False

        decided = rec["event"].wait(timeout_s if timeout_s is not None else self.timeout_s)
        with self._lock:
            self._pending.pop(confirm_id, None)
        if not decided:
            return False
        return bool(rec["approved"])

    def resolve(self, confirm_id: str, approved: bool) -> bool:
        """写入决定并唤醒等待方。返回是否找到了这个待确认项。"""
        with self._lock:
            rec = self._pending.get(confirm_id)
        if rec is None:
            return False
        rec["approved"] = bool(approved)
        rec["event"].set()
        return True

    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending)

    def pending_ids(self) -> list[str]:
        with self._lock:
            return list(self._pending)

    def clear(self) -> None:
        """仅测试用：把所有等待方按「拒绝」唤醒，避免测试间串味。"""
        with self._lock:
            recs = list(self._pending.values())
            self._pending.clear()
        for r in recs:
            r["approved"] = False
            r["event"].set()


__all__ = ["ConfirmationRegistry", "DEFAULT_TIMEOUT_S"]
