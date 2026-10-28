"""Flask backend for the real-time eye blink detection dashboard."""
import time

from flask import Flask, Response, jsonify, render_template

from camera import create_stream


def create_app(stream=None) -> Flask:
    app = Flask(__name__)
    stream = stream or create_stream()
    app.config["STREAM"] = stream

    @app.route("/")
    def index():
        return render_template("index.html")

    def mjpeg_generator():
        while True:
            try:
                frame = stream.get_jpeg()
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                time.sleep(0.03)
            except GeneratorExit:
                break
            except Exception:
                time.sleep(0.5)

    @app.route("/video_feed")
    def video_feed():
        return Response(mjpeg_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/api/status")
    def api_status():
        try:
            return jsonify(stream.get_status())
        except Exception as exc:
            return jsonify(error=str(exc), status="Backend Error", message=str(exc)), 500

    @app.route("/api/reset", methods=["POST"])
    def api_reset():
        try:
            return jsonify(success=True, count=stream.detector.reset())
        except Exception as exc:
            return jsonify(success=False, error=str(exc)), 500

    @app.errorhandler(404)
    def not_found(_):
        return jsonify(error="Not found"), 404

    @app.errorhandler(405)
    def bad_method(_):
        return jsonify(error="Method not allowed"), 405

    @app.errorhandler(500)
    def server_error(_):
        return jsonify(error="Internal server error"), 500

    return app


if __name__ == "__main__":
    print("Open http://127.0.0.1:5000 in your browser")
    # debug/reloader must stay off: the reloader starts a 2nd process that fights over the webcam
    create_app().run(host="127.0.0.1", port=5000, debug=False, threaded=True, use_reloader=False)
