"""Background capture thread: webcam (or demo video file) -> detection -> JPEG frames."""
import os
import sys
import threading
import time

import cv2
import numpy as np

from detector import BlinkDetector


class VideoStream:
    def __init__(self, source=0, backend=None):
        self.source = source
        self._is_file = isinstance(source, str)
        self.detector = BlinkDetector(backend)
        self._lock = threading.Lock()
        self._jpeg = None
        self._camera_ok = None            # None = still starting, False = error, True = running
        self._message = "Starting camera..."
        self._detector_error = None
        self._running = True
        self._placeholder = self._make_placeholder("Camera unavailable")
        threading.Thread(target=self._loop, daemon=True).start()

    # ---------- public ----------
    def get_jpeg(self) -> bytes:
        with self._lock:
            return self._jpeg if (self._jpeg and self._camera_ok) else self._placeholder

    def get_status(self) -> dict:
        snap = self.detector.snapshot()
        with self._lock:
            if self._camera_ok is None:
                status, message = "Starting Camera", self._message
            elif not self._camera_ok:
                status, message = "Camera Error", self._message
            elif self._detector_error:
                status, message = "Detector Error", self._detector_error
            elif snap["face_detected"]:
                status, message = "Face Detected - Monitoring", self._message
            else:
                status, message = "Face Not Detected", self._message
            snap.update(status=status, message=message, camera_ok=self._camera_ok)
        return snap

    def stop(self):
        self._running = False

    # ---------- internals ----------
    @staticmethod
    def _make_placeholder(text: str) -> bytes:
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(img, text, (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return cv2.imencode(".jpg", img)[1].tobytes()

    def _set(self, **kw):
        with self._lock:
            if "ok" in kw:
                self._camera_ok = kw["ok"]
            if "message" in kw:
                self._message = kw["message"]
            if "det_err" in kw:
                self._detector_error = kw["det_err"]

    def _open(self):
        if self._is_file:
            cap = cv2.VideoCapture(self.source)
        else:
            backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
            cap = cv2.VideoCapture(self.source, backend)
        if cap.isOpened():
            if not self._is_file:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            return cap
        cap.release()
        return None

    def _loop(self):
        cap, failures = None, 0
        while self._running:
            try:
                if cap is None:
                    cap = self._open()
                    if cap is None:
                        self._set(ok=False, message="Webcam unavailable, already in use, or permission "
                                                    "denied. Retrying...")
                        time.sleep(2)
                        continue
                    failures = 0

                ok, frame = cap.read()
                if not ok or frame is None:
                    if self._is_file:                       # demo video: loop it
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        failures += 1
                        if failures > 3:
                            cap.release(); cap = None; failures = 0
                        continue
                    failures += 1
                    if failures > 15:
                        self._set(ok=False, message="Lost connection to webcam. Reconnecting...")
                        cap.release(); cap = None
                    time.sleep(0.05)
                    continue
                failures = 0

                if not self._is_file:
                    frame = cv2.flip(frame, 1)              # mirror view for webcams
                try:
                    frame = self.detector.process(frame)
                    self._set(det_err=None)
                except Exception as exc:                    # detector problems must not kill the stream
                    self._set(det_err=str(exc))
                    time.sleep(0.2)

                done, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if done:
                    with self._lock:
                        self._jpeg = buf.tobytes()
                        self._camera_ok = True
                        self._message = "Camera running"
                if self._is_file:
                    time.sleep(1 / 30)
            except Exception as exc:                        # last-resort guard: reopen the camera
                self._set(ok=False, message=f"Video error: {exc}")
                if cap is not None:
                    cap.release(); cap = None
                time.sleep(1)
        if cap is not None:
            cap.release()


def create_stream(backend=None) -> VideoStream:
    video = os.environ.get("VIDEO_FILE")          # optional demo mode without a webcam
    if video:
        return VideoStream(video, backend)
    return VideoStream(int(os.environ.get("CAMERA_INDEX", 0)), backend)
