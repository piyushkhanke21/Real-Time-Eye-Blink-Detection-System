# Eye Blink Detection

A small Flask app that watches your webcam, counts how many times you blink, and shows the count next to the live video. There's a Reset button that sets the count back to zero without touching the camera or reloading the page.

I used OpenCV for the camera, MediaPipe for the face landmarks, and the Eye Aspect Ratio (EAR) to decide whether an eye is open or closed.

## Running it

I developed this on Windows with Python 3.11.5 in VS Code.

1. Open the project folder in VS Code.
2. Open a terminal (Terminal > New Terminal) and set up a virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Press Ctrl+Shift+P, choose "Python: Select Interpreter" and pick the one inside `venv`.
4. Start the app with `python app.py`.
5. Go to http://127.0.0.1:5000, allow the camera if the browser asks, and blink a few times.

A few things that tripped me up while testing:

- Only one program can use the webcam at a time. If you see "Camera Error", close Zoom, Teams, or any browser tab that has the camera open.
- To use a different camera: in PowerShell run `$env:CAMERA_INDEX=1; python app.py`.
- No webcam? You can point it at a video file instead: `VIDEO_FILE=path/to/video.mp4 python app.py`. It just loops the video.
- On Python 3.13+ (or if pip installs a very new MediaPipe) the old `mp.solutions` API is gone. Run `pip install -U mediapipe numpy opencv-python` and the code switches to the newer Tasks API on its own. That one downloads a small face model (about 4 MB) into `models/` the first time, so it needs internet once. If the download fails the dashboard shows the link and where to put the file.

## Tests

They don't need a webcam or internet:
```
python -m unittest discover -s tests -v
```

## Libraries

- Flask: web server, video stream, JSON endpoints
- opencv-python: reading the webcam, drawing the eye outlines, JPEG encoding
- mediapipe: face landmarks (468 points but we need basically two i.e two eyes)
- numpy: the distance maths for EAR

## Files

```
app.py          Flask app and routes (/, /video_feed, /api/status, POST /api/reset)
camera.py       thread that reads the camera, handles errors/reconnects, optional demo video
detector.py     EAR, blink logic, the counter
landmarks.py    wraps the two MediaPipe APIs (old solutions API / newer Tasks API)
templates/      index.html
static/         style.css, app.js
tests/          unit tests and an end-to-end test
```

## How the blink detection works

MediaPipe gives me six points around each eye: the two corners and two points on each of the upper and lower eyelids. From those:

```
EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
```

Open eyes sit around 0.25 to 0.35 and it drops to roughly 0.1 when the eye closes. I average the left and right eye so one noisy eye doesn't cause trouble.

My first version used a fixed cutoff of 0.2, and it didn't work well. It missed blinks on some faces and fired constantly on others, depending on eye shape, distance from the camera and glasses. So now the first 10 frames are used to learn what "open" looks like for the person in front of the camera, and that baseline keeps adjusting slowly (only from frames where the eyes are open). The eye counts as closed when EAR falls below 75% of the baseline, and as open again when it climbs back above 82%. Having two different numbers stops the value from flickering when it hovers near a single cutoff.

The counting itself is a tiny state machine: OPEN, then CLOSED, then OPEN again. The counter goes up only on that last step, which is why holding your eyes shut doesn't keep adding blinks. Two extra rules apply:

- The eyes must stay closed for at least 2 frames, to ignore one-frame glitches.
- A closure longer than 1 second isn't counted, since that's more likely someone closing their eyes than blinking.

If the face disappears while the eyes are closed, nothing is counted. A short tracking loss (up to 8 frames) is ignored before the status changes to "Face Not Detected".

## How the pieces fit together

A background thread owns the camera. It keeps reading frames, running detection and encoding JPEGs, no matter whether a browser is connected. Flask just serves whatever the latest frame and state are.

- Video: `/video_feed` streams the frames as MJPEG, so the page only needs a plain `<img>` tag.
- Count and status: the page asks `/api/status` every 250 ms with `fetch` and updates the numbers. No page reloads.
- Reset: the button sends `POST /api/reset`, which zeroes the counter under a lock. The camera thread isn't stopped, so the video and detection carry on, and if you're mid-blink when you press it, that blink still finishes properly.

## What happens when things go wrong

- Camera missing, busy, or permission denied: status shows "Camera Error", a placeholder image is shown, and it retries every 2 seconds.
- Camera unplugged while running: it's released and reopened automatically.
- Face not visible: "Face Not Detected" after the short grace period, with no false blinks.
- Face model can't load: status shows "Detector Error" with a message, the video keeps streaming, and it retries every 5 seconds.
- Reset fails: an error message appears under the button and the app keeps running.
- Server unreachable: the dashboard shows "Backend Error", keeps retrying, and reconnects the video stream when the server is back.
- Anything unexpected in the capture loop is caught and the camera is set up again, instead of the whole app dying.

## Decisions and problems I ran into

- MediaPipe instead of dlib. dlib needs a C++ compiler on Windows and a ~100 MB model file, and installing it is a hassle. MediaPipe is one pip install and handled head turns and lighting better in my tests.
- Support for two MediaPipe APIs. Newer MediaPipe releases removed `mp.solutions`, which broke the first version on a fresh install. `landmarks.py` now checks which one is available and uses it.
- Camera on the server, not in the browser. The task asks for a Python backend, and this keeps all the detection logic in Python.
- The fixed EAR threshold. Covered above, the adaptive baseline fixed it.
- Double counting. Checking each frame on its own counted one blink several times. The state machine fixed that.
- The Flask debug reloader. It starts a second process, and both processes tried to open the webcam. I turned the reloader off.
- Sharing state between threads. The camera thread and Flask's request threads read and write the same values, so I protected them with locks.
- Testing without a camera. The landmark source can be swapped out, so the tests feed in made-up EAR sequences and a generated video to check the blink logic, the API, the stream, reset and the error paths. 
Also used MediaPipe Face Landmarker, developed by Google (the MediaPipe team at Google AI Edge). It’s a pre-trained model, so I didn’t train anything. The file face_landmarker.task is a bundle of three models i.e Face detection, Face mesh, Blendshape prediction.

## Limitations

- Only one face is tracked.
- Very dim light or something covering the eyes (hand, thick glasses glare) can make tracking drop.
- The frame-count rules assume the video runs at roughly 15 to 30 FPS. On a very slow machine the numbers might need tuning.
- Flask's built-in server is fine for a demo, but I wouldn't deploy it as is.