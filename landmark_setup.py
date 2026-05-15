"""
landmark_setup.py — Interactive Landmark Path Builder
======================================================
Guides you step-by-step through capturing landmark photos from your camera
to define a navigation path for the drone.

Usage:
  python landmark_setup.py --output landmarks/mission_01

Controls (shown in the live preview window):
  SPACE  → Capture current frame as the next landmark
  D      → Delete last captured landmark
  Q      → Quit and finalize the path

Output structure:
  landmarks/mission_01/
      00_landmark.jpg
      01_landmark.jpg
      02_landmark.jpg
      ...  (as many as you capture)
      path_manifest.json   ← metadata file describing the path
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

import cv2


# ── Terminal color helpers ─────────────────────────────────────────────────────

def _c(text: str, code: str) -> str:
    """ANSI color wrap (gracefully disabled on Windows without VT support)."""
    try:
        return f"\033[{code}m{text}\033[0m"
    except Exception:
        return text

GREEN  = lambda t: _c(t, "92")
YELLOW = lambda t: _c(t, "93")
CYAN   = lambda t: _c(t, "96")
RED    = lambda t: _c(t, "91")
BOLD   = lambda t: _c(t, "1")


# ── OSD overlay ───────────────────────────────────────────────────────────────

def _draw_overlay(frame, captured: int, mission_name: str):
    """Draws an on-screen display on the camera preview frame."""
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Semi-transparent top bar
    cv2.rectangle(overlay, (0, 0), (w, 80), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    font   = cv2.FONT_HERSHEY_SIMPLEX
    green  = (100, 230, 100)
    white  = (240, 240, 240)
    yellow = (50, 200, 255)

    cv2.putText(frame, f"Mission: {mission_name}", (10, 28), font, 0.7, white, 2)
    cv2.putText(frame, f"Captured: {captured:02d}", (10, 58), font, 0.65, green, 2)
    cv2.putText(frame, "SPACE=Capture  D=Delete last  Q=Quit",
                (w - 440, h - 14), font, 0.52, yellow, 1)

    # Crosshair guide
    cx, cy = w // 2, h // 2
    cv2.line(frame, (cx - 30, cy), (cx + 30, cy), (100, 230, 100), 1)
    cv2.line(frame, (cx, cy - 30), (cx, cy + 30), (100, 230, 100), 1)
    cv2.circle(frame, (cx, cy), 50, (100, 230, 100), 1)

    return frame


# ── Flash effect ──────────────────────────────────────────────────────────────

def _flash(win: str, frame, duration_ms: int = 120):
    """Briefly shows a white flash to simulate a camera shutter."""
    white = frame.copy()
    white[:] = (255, 255, 255)
    cv2.imshow(win, white)
    cv2.waitKey(duration_ms)


# ── Main capture loop ─────────────────────────────────────────────────────────

def run_capture(output_dir: Path, camera_index: int, mission_name: str):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(RED(f"ERROR: Cannot open camera at index {camera_index}."))
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    win = "Landmark Setup — Press SPACE to capture"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 960, 600)

    captured_paths = []

    print()
    print(BOLD("━" * 60))
    print(BOLD("  🛸 Landmark Path Builder"))
    print(BOLD("━" * 60))
    print(f"  Mission   : {CYAN(mission_name)}")
    print(f"  Output    : {CYAN(str(output_dir))}")
    print(f"  Camera    : index {camera_index}")
    print()
    print(f"  {YELLOW('How to use:')}")
    print("   1. Walk to each landmark location")
    print("   2. Point camera at the distinctive feature you want the")
    print("      drone to recognize (doorway, shelf, poster, etc.)")
    print("   3. Press SPACE to capture")
    print("   4. Repeat for every waypoint along the path")
    print("   5. The LAST photo will be the goal (landing spot)")
    print("   6. Press Q when done")
    print()
    print(BOLD("━" * 60))
    print()

    while True:
        ret, frame = cap.read()
        if not ret:
            print(RED("Camera read failed. Exiting."))
            break

        display = frame.copy()
        display = _draw_overlay(display, len(captured_paths), mission_name)
        cv2.imshow(win, display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):  # SPACE → capture
            idx       = len(captured_paths)
            filename  = f"{idx:02d}_landmark.jpg"
            save_path = output_dir / filename

            cv2.imwrite(str(save_path), frame)
            _flash(win, display)

            captured_paths.append(save_path)
            role = "🏁 FINAL GOAL" if idx == 0 else f"Waypoint {idx:02d}"  # will update on quit
            print(f"  [{idx:02d}] Captured → {filename}  ({role})")

        elif key == ord('d') or key == ord('D'):  # D → delete last
            if captured_paths:
                last = captured_paths.pop()
                try:
                    last.unlink()
                except FileNotFoundError:
                    pass
                print(f"  {YELLOW('⚠')} Deleted {last.name}")
            else:
                print(f"  {YELLOW('Nothing to delete.')}")

        elif key == ord('q') or key == ord('Q'):  # Q → quit
            break

    cap.release()
    cv2.destroyAllWindows()

    return captured_paths


# ── Post-capture: rename and manifest ─────────────────────────────────────────

def finalize(output_dir: Path, captured_paths: list, mission_name: str):
    if not captured_paths:
        print(YELLOW("\nNo landmarks captured. Exiting without saving."))
        return

    print()
    print(BOLD("━" * 60))
    print(BOLD("  Mission Summary"))
    print(BOLD("━" * 60))

    manifest = {
        "mission_name": mission_name,
        "created_at":   time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_waypoints": len(captured_paths),
        "waypoints": [],
    }

    for i, path in enumerate(captured_paths):
        role = "final_goal" if i == len(captured_paths) - 1 else "waypoint"
        entry = {
            "index":    i,
            "filename": path.name,
            "role":     role,
        }
        manifest["waypoints"].append(entry)
        icon = "🏁" if role == "final_goal" else f"[{i:02d}]"
        print(f"  {icon}  {path.name}  ← {role}")

    manifest_path = output_dir / "path_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print()
    print(f"  {GREEN('✅ Path saved to:')} {output_dir}")
    print(f"  {GREEN('✅ Manifest     :')} {manifest_path.name}")
    print()
    print(BOLD("  ── How to fly this path ───────────────────────────────"))
    print()
    print(f"  python landmark_flight.py \\")
    print(f"      --landmarks {output_dir} \\")
    print(f"      --dry-run")
    print()
    print(f"  python landmark_flight.py \\")
    print(f"      --landmarks {output_dir} \\")
    print(f"      --address udp://:14540 \\")
    print(f"      --altitude 1.5")
    print()
    print(BOLD("━" * 60))


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Interactive landmark path builder — capture photos to define a navigation path."
    )
    parser.add_argument(
        "--output", "-o",
        default="landmarks/mission_01",
        help="Output directory for landmark images (default: landmarks/mission_01)"
    )
    parser.add_argument(
        "--camera", "-c",
        type=int, default=0,
        help="OpenCV camera index (default: 0)"
    )
    parser.add_argument(
        "--name", "-n",
        default=None,
        help="Mission name (default: derived from output directory name)"
    )

    args    = parser.parse_args()
    out_dir = Path(args.output)
    name    = args.name or out_dir.name

    captured = run_capture(out_dir, args.camera, name)
    finalize(out_dir, captured, name)


if __name__ == "__main__":
    main()
