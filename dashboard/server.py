"""
dashboard/server.py — Live Flight Monitoring Dashboard Backend
==============================================================
Lightweight Flask server that serves the dashboard HTML and exposes
a /telemetry endpoint that the frontend polls every 500ms.

Run:
  python dashboard/server.py          (default port 5050)
  python dashboard/server.py --port 8080
"""

import json
import time
import argparse
from pathlib import Path

try:
    from flask import Flask, jsonify, send_from_directory
except ImportError:
    print("ERROR: Flask not installed. Run: pip install flask")
    raise

TELEM_FILE = Path("logs/live_telemetry.json")
app = Flask(__name__, static_folder="static")


@app.route("/")
def index():
    return send_from_directory(Path(__file__).parent, "index.html")


@app.route("/telemetry")
def telemetry():
    if TELEM_FILE.exists():
        try:
            with open(TELEM_FILE) as f:
                data = json.load(f)
            data["server_ts"] = time.time()
            return jsonify(data)
        except Exception as e:
            return jsonify({"error": str(e), "server_ts": time.time()}), 500
    return jsonify({
        "status": "waiting",
        "message": "No telemetry yet — start landmark_flight.py",
        "server_ts": time.time()
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5050)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()
    print(f"[Dashboard] http://localhost:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)
