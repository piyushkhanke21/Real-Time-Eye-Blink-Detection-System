"""Blink detection: Eye Aspect Ratio (EAR) + an open -> closed -> open state machine."""
import threading
import time

import cv2
import numpy as np

from landmarks import create_backend


def eye_aspect_ratio(pts) -> float:
    """EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)."""
    p1, p2, p3, p4, p5, p6 = [np.asarray(p, dtype=float) for p in pts]
    horizontal = np.linalg.norm(p1 - p4)
    if horizontal == 0:
        return 0.0
    return float((np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)) / (2.0 * horizontal))


class BlinkDetector:
    WARMUP_FRAMES = 10        # frames used to learn the user's open-eye EAR
    CLOSE_RATIO = 0.75        # closed when EAR < baseline * 0.75
    REOPEN_RATIO = 0.82       # reopened when EAR > baseline * 0.82 (hysteresis)
    MIN_THRESHOLD = 0.10
    MIN_CLOSED_FRAMES = 2     # ignore single-frame noise
    MAX_BLINK_SECONDS = 1.0   # longer closure = eyes shut, not a blink
    FACE_LOSS_GRACE = 8       # frames of tracking loss tolerated

    def __init__(self, backend=None):
        self._backend = backend
        self._backend_error = None
        self._next_backend_try = 0.0
        self._lock = threading.Lock()
        self._count = 0
        self._baseline = None
        self._warmup = []
        self._eyes_closed = False
        self._closed_frames = 0
        self._closed_since = 0.0
        self._lost_frames = 0
        self._face_detected = False
        self._ear = 0.0

    # ---------- public ----------
    def reset(self) -> int:
        """Zero the counter. Detection state is kept so blinking continues seamlessly."""
        with self._lock:
            self._count = 0
            return self._count

    def snapshot(self) -> dict:
        with self._lock:
            return {"count": self._count, "face_detected": self._face_detected,
                    "eyes_closed": self._eyes_closed, "ear": round(self._ear, 3)}

    def process(self, frame_bgr):
        """Analyse a BGR frame, update blink state, return the annotated frame."""
        backend = self._get_backend()
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        eyes = backend.find_eyes(rgb)
        if eyes is None:
            self._on_face_lost()
            return frame_bgr
        left, right = eyes
        ear = (eye_aspect_ratio(left) + eye_aspect_ratio(right)) / 2.0
        self._update(ear)
        self._draw(frame_bgr, left, right, ear)
        return frame_bgr

    # ---------- internals ----------
    def _get_backend(self):
        if self._backend is None:
            now = time.time()
            if now < self._next_backend_try:
                raise RuntimeError(self._backend_error)
            try:
                self._backend = create_backend()
            except Exception as exc:
                self._backend_error = f"Face model failed to load: {exc}"
                self._next_backend_try = now + 5
                raise RuntimeError(self._backend_error) from exc
        return self._backend

    def _on_face_lost(self):
        with self._lock:
            self._lost_frames += 1
            if self._lost_frames > self.FACE_LOSS_GRACE:
                self._face_detected = False
                self._eyes_closed = False     # losing the face must never count as a blink
                self._closed_frames = 0

    def _update(self, ear: float, now: float = None):
        now = time.time() if now is None else now
        with self._lock:
            self._lost_frames = 0
            self._face_detected = True
            self._ear = ear

            if self._baseline is None:        # calibration warm-up, no counting
                self._warmup.append(ear)
                if len(self._warmup) >= self.WARMUP_FRAMES:
                    self._baseline = float(np.percentile(self._warmup, 75))
                return

            close_thr = max(self.MIN_THRESHOLD, self._baseline * self.CLOSE_RATIO)
            open_thr = max(self.MIN_THRESHOLD, self._baseline * self.REOPEN_RATIO)

            if not self._eyes_closed:
                if ear < close_thr:
                    self._eyes_closed = True
                    self._closed_frames = 1
                    self._closed_since = now
                elif ear > 0.15:              # adapt baseline on open-eye frames only
                    self._baseline = 0.98 * self._baseline + 0.02 * ear
            elif ear > open_thr:              # eyes reopened -> blink complete
                if (self._closed_frames >= self.MIN_CLOSED_FRAMES
                        and now - self._closed_since <= self.MAX_BLINK_SECONDS):
                    self._count += 1
                self._eyes_closed = False
                self._closed_frames = 0
            else:
                self._closed_frames += 1      # still closed: no double counting

    def _draw(self, frame, left, right, ear):
        color = (0, 0, 255) if self._eyes_closed else (0, 220, 0)
        for eye in (left, right):
            cv2.polylines(frame, [eye.astype(np.int32)], True, color, 1, cv2.LINE_AA)
        cv2.putText(frame, f"EAR: {ear:.2f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
