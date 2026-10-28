"""Automated tests (no webcam or internet needed).  Run:  python -m unittest discover -s tests -v"""
import collections
import os
import sys
import tempfile
import time
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app                      # noqa: E402
from camera import VideoStream                  # noqa: E402
from detector import BlinkDetector, eye_aspect_ratio  # noqa: E402

OPEN, CLOSED = 0.30, 0.08


def eye_with_ear(ear, x0=100.0, y0=100.0):
    """Six eye points whose EAR equals `ear` (h = 5*ear for a 10px-wide eye)."""
    h = 5 * ear
    return np.array([[x0, y0], [x0 + 3, y0 - h], [x0 + 7, y0 - h],
                     [x0 + 10, y0], [x0 + 7, y0 + h], [x0 + 3, y0 + h]])


class FakeBackend:
    """Stands in for MediaPipe: EAR comes from a queue, so blinks are scripted."""
    def __init__(self):
        self.queue = collections.deque()
        self.face_visible = True

    def queue_blink(self, closed_frames=4, n=1):
        for _ in range(n):
            self.queue.extend([OPEN] * 8 + [CLOSED] * closed_frames + [OPEN] * 8)

    def find_eyes(self, rgb):
        if not self.face_visible:
            return None
        ear = self.queue.popleft() if self.queue else OPEN
        return eye_with_ear(ear), eye_with_ear(ear, x0=200.0)


def calibrated():
    d = BlinkDetector(backend=FakeBackend())
    t = 0.0
    for _ in range(15):
        d._update(OPEN, now=t); t += 0.033
    return d, t


class TestEAR(unittest.TestCase):
    def test_known_geometry(self):
        self.assertAlmostEqual(eye_aspect_ratio(eye_with_ear(0.3)), 0.3, places=6)
        self.assertAlmostEqual(eye_aspect_ratio(eye_with_ear(0.1)), 0.1, places=6)

    def test_degenerate_eye(self):
        self.assertEqual(eye_aspect_ratio(np.zeros((6, 2))), 0.0)


class TestBlinkStateMachine(unittest.TestCase):
    def run_seq(self, d, t, seq, dt=0.033):
        for ear in seq:
            d._update(ear, now=t); t += dt
        return t

    def test_no_count_during_warmup(self):
        d = BlinkDetector(backend=FakeBackend())
        self.run_seq(d, 0.0, [OPEN] * 3 + [CLOSED] * 3 + [OPEN] * 3)
        self.assertEqual(d.snapshot()["count"], 0)

    def test_single_blink_counts_once(self):
        d, t = calibrated()
        self.run_seq(d, t, [OPEN] * 3 + [CLOSED] * 4 + [OPEN] * 3)
        self.assertEqual(d.snapshot()["count"], 1)

    def test_eyes_held_closed_not_double_counted(self):
        d, t = calibrated()
        t = self.run_seq(d, t, [CLOSED] * 15)             # 0.5 s closed
        self.assertEqual(d.snapshot()["count"], 0)        # not counted until reopened
        self.run_seq(d, t, [OPEN] * 3)
        self.assertEqual(d.snapshot()["count"], 1)        # exactly one

    def test_single_frame_dip_is_noise(self):
        d, t = calibrated()
        self.run_seq(d, t, [OPEN] * 2 + [CLOSED] + [OPEN] * 3)
        self.assertEqual(d.snapshot()["count"], 0)

    def test_long_closure_is_not_a_blink(self):
        d, t = calibrated()
        t = self.run_seq(d, t, [CLOSED] * 4, dt=0.5)      # 2 s shut
        self.run_seq(d, t, [OPEN] * 3)
        self.assertEqual(d.snapshot()["count"], 0)

    def test_multiple_blinks(self):
        d, t = calibrated()
        for _ in range(5):
            t = self.run_seq(d, t, [OPEN] * 4 + [CLOSED] * 3 + [OPEN] * 4)
        self.assertEqual(d.snapshot()["count"], 5)

    def test_slow_and_fast_blinks(self):
        d, t = calibrated()
        t = self.run_seq(d, t, [OPEN] * 3 + [0.2, CLOSED, 0.2] + [OPEN] * 3)       # fast, partial
        t = self.run_seq(d, t, [CLOSED] * 20 + [OPEN] * 3, dt=0.03)                # slow ~0.6 s
        self.assertEqual(d.snapshot()["count"], 2)

    def test_face_loss_never_counts(self):
        d, t = calibrated()
        t = self.run_seq(d, t, [CLOSED] * 3)
        for _ in range(12):
            d._on_face_lost()
        self.assertFalse(d.snapshot()["face_detected"])
        self.run_seq(d, t, [OPEN] * 5)
        self.assertEqual(d.snapshot()["count"], 0)

    def test_brief_tracking_loss_keeps_face_status(self):
        d, t = calibrated()
        for _ in range(3):
            d._on_face_lost()
        self.assertTrue(d.snapshot()["face_detected"])

    def test_reset_then_count_from_zero(self):
        d, t = calibrated()
        t = self.run_seq(d, t, [CLOSED] * 3 + [OPEN] * 3)
        self.assertEqual(d.snapshot()["count"], 1)
        self.assertEqual(d.reset(), 0)
        self.run_seq(d, t, [CLOSED] * 3 + [OPEN] * 3)
        self.assertEqual(d.snapshot()["count"], 1)

    def test_reset_during_closed_eyes_keeps_detection_working(self):
        d, t = calibrated()
        t = self.run_seq(d, t, [CLOSED] * 3)
        d.reset()
        self.run_seq(d, t, [OPEN] * 3)
        self.assertEqual(d.snapshot()["count"], 1)        # that blink completes after reset


def make_video(path, frames=60):
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 30, (640, 480))
    assert w.isOpened(), "cannot create test video"
    for i in range(frames):
        img = np.full((480, 640, 3), 60, np.uint8)
        cv2.circle(img, (100 + i * 5, 240), 40, (200, 200, 50), -1)
        w.write(img)
    w.release()


def wait_for(cond, timeout=10.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.05)
    return False


class TestEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.video = os.path.join(cls.tmp, "demo.avi")
        make_video(cls.video)
        cls.backend = FakeBackend()
        cls.stream = VideoStream(cls.video, cls.backend)
        cls.client = create_app(cls.stream).test_client()

    @classmethod
    def tearDownClass(cls):
        cls.stream.stop()

    def status(self):
        return self.client.get("/api/status").get_json()

    def test_1_index_and_static(self):
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn(b"Reset Counter", r.data)
        self.assertIn(b"video_feed", r.data)
        self.assertEqual(self.client.get("/static/app.js").status_code, 200)
        self.assertEqual(self.client.get("/static/style.css").status_code, 200)

    def test_2_video_feed_streams_jpeg(self):
        wait_for(lambda: self.status()["camera_ok"])
        r = self.client.get("/video_feed")
        self.assertIn("multipart/x-mixed-replace", r.content_type)
        chunk = next(iter(r.response))
        r.close()
        self.assertTrue(chunk.startswith(b"--frame"))
        self.assertIn(b"\xff\xd8", chunk)                 # JPEG magic bytes

    def test_3_blinks_counted_live_and_status(self):
        self.assertTrue(wait_for(lambda: self.status()["face_detected"]))
        self.assertIn("Monitoring", self.status()["status"])
        self.backend.queue_blink(n=3)
        self.assertTrue(wait_for(lambda: self.status()["count"] == 3), self.status())
        time.sleep(0.5)
        self.assertEqual(self.status()["count"], 3)       # no extra counts while eyes open

    def test_4_reset_keeps_everything_running(self):
        self.backend.queue_blink(n=2)
        self.assertTrue(wait_for(lambda: self.status()["count"] >= 2))
        r = self.client.post("/api/reset")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json(), {"success": True, "count": 0})
        self.assertEqual(self.status()["count"], 0)
        self.assertTrue(self.status()["camera_ok"])       # camera uninterrupted
        self.backend.queue_blink(n=1)
        self.assertTrue(wait_for(lambda: self.status()["count"] == 1), self.status())  # counts from zero

    def test_5_face_not_detected_then_recovers(self):
        self.backend.face_visible = False
        self.assertTrue(wait_for(lambda: self.status()["status"] == "Face Not Detected"))
        self.backend.face_visible = True
        self.assertTrue(wait_for(lambda: "Monitoring" in self.status()["status"]))

    def test_6_bad_requests_handled(self):
        self.assertEqual(self.client.get("/api/reset").status_code, 405)
        self.assertEqual(self.client.get("/nope").status_code, 404)


class TestErrorPaths(unittest.TestCase):
    def test_camera_unavailable(self):
        s = VideoStream(99, FakeBackend())
        client = create_app(s).test_client()
        self.assertEqual(client.get("/api/status").get_json()["status"], "Starting Camera")
        self.assertTrue(wait_for(lambda: client.get("/api/status").get_json()["status"] == "Camera Error", 5))
        st = client.get("/api/status").get_json()
        self.assertFalse(st["camera_ok"])
        self.assertIn("unavailable", st["message"].lower())
        self.assertEqual(client.post("/api/reset").get_json()["count"], 0)   # reset still works
        r = client.get("/video_feed")
        self.assertIn(b"\xff\xd8", next(iter(r.response)))                    # placeholder frame
        r.close(); s.stop()

    def test_detector_failure_does_not_kill_stream(self):
        class Boom:
            def find_eyes(self, rgb):
                raise RuntimeError("model exploded")
        tmp = tempfile.mkdtemp(); path = os.path.join(tmp, "v.avi"); make_video(path)
        s = VideoStream(path, Boom())
        client = create_app(s).test_client()
        self.assertTrue(wait_for(lambda: client.get("/api/status").get_json()["status"] == "Detector Error", 5))
        self.assertIn("model exploded", client.get("/api/status").get_json()["message"])
        self.assertTrue(client.get("/api/status").get_json()["camera_ok"])
        s.stop()


if __name__ == "__main__":
    unittest.main(verbosity=2)
