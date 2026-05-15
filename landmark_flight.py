"""
landmark_flight.py — Landmark-Chain Navigation Entry Point
==========================================================
Navigates the drone through an ordered sequence of landmark photos.

Workflow:
  1. Load landmark images from a directory (sorted by filename)
  2. Encode each into an embedding via GoalEncoder
  3. Fly toward landmark[0], then [1], then [2] ... until final → LAND

Usage:
  # Dry-run (camera only, no drone)
  python landmark_flight.py --landmarks landmarks/mission_01 --dry-run

  # Real flight
  python landmark_flight.py --landmarks landmarks/mission_01 --address udp://:14540

  # With live dashboard
  python landmark_flight.py --landmarks landmarks/mission_01 --dry-run --dashboard
"""

import os
import sys
import asyncio
import argparse
import time
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent))

from drone_nav.utils.device import get_device
from drone_nav.perception.encoders import (
    PerceptionBackbone, VisualEncoder, GoalEncoder, DepthEncoder, MemoryModule
)
from drone_nav.nav.path_follower import PathFollower
from drone_nav.nav.goal_matcher import GoalMatcher
from drone_nav.nav.landmark_sequencer import LandmarkSequencer
from drone_nav.control.planner import IntegratedPlanner
from drone_navigator.controller import DroneController

# ── Constants ─────────────────────────────────────────────────────────────────
WEIGHTS_PATH      = "checkpoints/nav_stack_v2_2.pth"
CONTROL_HZ        = 10
MAX_VELOCITY      = 0.5
EMA_LAMBDA        = 0.85
MAX_STEPS         = 600          # longer budget for multi-waypoint missions
VPR_WINDOW        = 10
TAKEOFF_ALTITUDE  = 1.5
HEARTBEAT_HZ      = 5
WATCHDOG_LIMIT_S  = 1.5
OVERRUN_STREAK_MAX = 5
BATT_WARN_PCT     = 30.0
BATT_CRITICAL_PCT = 20.0
BATT_EMERGENCY_PCT = 10.0

# Landmark-specific
WAYPOINT_THRESHOLD = 0.78   # confidence to advance to next landmark
GOAL_THRESHOLD     = 0.83   # confidence to land at final landmark
CONFIRM_FRAMES     = 4      # consecutive frames needed before advancing


# ── Shared flight state ───────────────────────────────────────────────────────
class FlightState:
    def __init__(self):
        self.last_velocity        = {"vx": 0.0, "vy": 0.0}
        self.last_update_ts       = time.monotonic()
        self.stop                 = False
        self.consecutive_overruns = 0
        self.control_hz           = CONTROL_HZ
        self.battery_pct          = 100.0
        self.low_battery          = False
        # Dashboard telemetry (written each step, read by dashboard server)
        self.telemetry: dict      = {}

_state = FlightState()

# ── Pre-processing ────────────────────────────────────────────────────────────
PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ── Heartbeat task (C1) ───────────────────────────────────────────────────────
async def heartbeat_task(controller, args):
    while not _state.stop:
        await asyncio.sleep(1.0 / HEARTBEAT_HZ)
        if args.dry_run:
            continue
        stale_s = time.monotonic() - _state.last_update_ts
        if stale_s > WATCHDOG_LIMIT_S:
            print(f"[WATCHDOG] Stale {stale_s:.2f}s — emergency hover.")
            await controller.emergency_hover()
            continue
        await controller.move_to(_state.last_velocity)


# ── Battery monitor task (C2) ─────────────────────────────────────────────────
async def battery_monitor_task(controller, args):
    while not _state.stop:
        await asyncio.sleep(2.0)
        if args.dry_run:
            continue
        batt = await controller.get_battery()
        if batt < 0:
            continue
        _state.battery_pct = batt
        if batt < BATT_EMERGENCY_PCT:
            print(f"[BATT] EMERGENCY {batt:.1f}% — landing now.")
            await controller.emergency_hover()
            await asyncio.sleep(0.5)
            await controller.land()
            _state.stop = True
            return
        elif batt < BATT_CRITICAL_PCT:
            if not _state.low_battery:
                print(f"[BATT] CRITICAL {batt:.1f}% — controlled landing flagged.")
                _state.low_battery = True
        elif batt < BATT_WARN_PCT:
            print(f"[BATT] WARNING {batt:.1f}%.")


# ── Velocity helpers ──────────────────────────────────────────────────────────
def clip_velocity(velocity: list, max_v: float) -> list:
    result = []
    for v in velocity:
        if not isinstance(v, (int, float)) or v != v:  # NaN check (M5)
            print(f"[WARN] NaN in velocity — substituting 0.")
            return [0.0, 0.0, 0.0]
        result.append(max(-max_v, min(max_v, v)))
    return result


def frame_to_tensor(rgb_frame: np.ndarray, device) -> torch.Tensor:
    pil_img = Image.fromarray(rgb_frame)
    return PREPROCESS(pil_img).unsqueeze(0).to(device)


# ── Model loader ──────────────────────────────────────────────────────────────
def load_model(device):
    backbone      = PerceptionBackbone(architecture='resnet18').to(device)
    visual_enc    = VisualEncoder(backbone, use_netvlad=True).to(device)
    goal_enc      = GoalEncoder(backbone).to(device)
    depth_enc     = DepthEncoder(backbone).to(device)
    memory        = MemoryModule(input_dim=visual_enc.output_dim).to(device)
    path_follower = PathFollower(input_dim=memory.hidden_dim).to(device)
    goal_matcher  = GoalMatcher(input_dim=backbone.out_channels).to(device)

    if os.path.exists(WEIGHTS_PATH):
        print(f"[Model] Loading weights: {WEIGHTS_PATH}")
        ckpt = torch.load(WEIGHTS_PATH, map_location=device, weights_only=True)
        backbone.load_state_dict(ckpt.get('backbone', {}), strict=False)
        visual_enc.load_state_dict(ckpt.get('visual_encoder', {}), strict=False)
        goal_enc.load_state_dict(ckpt.get('goal_encoder', {}), strict=False)
        depth_enc.load_state_dict(ckpt.get('depth_encoder', {}), strict=False)
        path_follower.load_state_dict(ckpt.get('path_follower', {}), strict=False)
        goal_matcher.load_state_dict(ckpt.get('goal_matcher', {}), strict=False)
        if 'memory' in ckpt:
            memory.load_state_dict(ckpt['memory'])
    else:
        print(f"[WARNING] No checkpoint at '{WEIGHTS_PATH}'. Running untrained model.")

    for m in [backbone, visual_enc, goal_enc, depth_enc, memory, path_follower, goal_matcher]:
        m.eval()

    return backbone, visual_enc, goal_enc, depth_enc, memory, path_follower, goal_matcher


# ── CSV logger ────────────────────────────────────────────────────────────────
def make_csv_logger(landmark_dir: Path):
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    log_path = logs_dir / f"landmark_flight_{int(time.time())}.csv"
    f = open(log_path, 'w', newline='')
    writer = csv.writer(f)
    writer.writerow([
        'step', 'timestamp', 'waypoint_idx', 'waypoint_name',
        'vx', 'vy', 'goal_conf', 'repulse', 'action',
        'progress_pct', 'battery_pct', 'loop_ms'
    ])
    print(f"[Log] Flight log: {log_path}")
    return f, writer, log_path


# ── Dashboard telemetry writer ────────────────────────────────────────────────
_TELEM_FILE = Path("logs/live_telemetry.json")

def write_telemetry(data: dict):
    """Write current step telemetry to a JSON file for the dashboard to read."""
    try:
        _TELEM_FILE.parent.mkdir(exist_ok=True)
        with open(_TELEM_FILE, 'w') as f:
            json.dump(data, f)
    except Exception:
        pass


# ── Core flight loop ──────────────────────────────────────────────────────────
async def landmark_flight_loop(
    args, controller, planner, sequencer: LandmarkSequencer,
    visual_enc, goal_enc, depth_enc, goal_matcher, device
):
    """
    Perception → Planning → Landmark Matching → Actuation loop.
    Advances through waypoints in order; lands only at the final landmark.
    """
    vpr_memory = []
    step = 0

    print("\n[Flight] Starting landmark navigation loop.")
    print(f"         Waypoints: {sequencer.total_waypoints} | "
          f"Max steps: {MAX_STEPS} | Hz: {_state.control_hz}\n")

    log_f, log_writer, log_path = make_csv_logger(Path(args.landmarks))

    try:
        while step < MAX_STEPS and not _state.stop:
            loop_budget_s = 1.0 / _state.control_hz
            t_start = time.monotonic()

            # ── A. Perception ─────────────────────────────────────────────────
            rgb_frame = await controller.get_camera_frame()
            tensor    = frame_to_tensor(rgb_frame, device)

            try:
                with torch.no_grad():
                    vpr = visual_enc(tensor)
                    vpr_memory.append(vpr)
                    if len(vpr_memory) > VPR_WINDOW:
                        vpr_memory.pop(0)
                    vpr_seq = torch.stack(vpr_memory, dim=1)
                    if vpr_seq.shape[1] < VPR_WINDOW:
                        pad = torch.zeros(1, VPR_WINDOW - vpr_seq.shape[1],
                                          vpr_seq.shape[2], device=device)
                        vpr_seq = torch.cat([pad, vpr_seq], dim=1)

                    obs_v = goal_enc(tensor)
                    depth = depth_enc(tensor)
            except Exception as e:
                print(f"[WARN] Inference error step {step}: {e}. Hovering.")
                if not args.dry_run:
                    await controller.emergency_hover()
                step += 1
                continue

            # ── B. Landmark sequencing (replaces single-goal matching) ────────
            seq_result = sequencer.step(goal_matcher, obs_v)
            lm_action  = seq_result['action']         # MOVE | ADVANCE | LAND
            conf       = seq_result['goal_confidence']
            wp_idx     = seq_result['active_idx']
            wp_name    = seq_result['active_name']
            progress   = seq_result['progress_pct']

            # ── C. Path planning (velocity toward current landmark) ───────────
            plan_result = planner.plan(
                obs_v, vpr_seq,
                sequencer.active_embedding,   # ← active landmark, not static goal
                depth_map=depth, vpr_obs=vpr
            )
            velocity   = clip_velocity(plan_result['velocity'], MAX_VELOCITY)
            repulse    = plan_result.get('repulsive_active', False)

            # ── D. Logging ────────────────────────────────────────────────────
            vx, vy, _ = velocity
            elapsed   = time.monotonic() - t_start

            print(
                f"[Step {step:03d}] WP[{wp_idx:02d}/{sequencer.total_waypoints-1:02d}] "
                f"'{wp_name[:18]}' | "
                f"conf={conf:.3f} | vx={vx:+.3f} vy={vy:+.3f} | "
                f"{'⚠ REPULSE' if repulse else 'clear'} | "
                f"progress={progress*100:.0f}% | {elapsed*1000:.0f}ms"
            )

            log_writer.writerow([
                step, f"{time.time():.3f}", wp_idx, wp_name,
                f"{vx:.4f}", f"{vy:.4f}", f"{conf:.4f}",
                int(repulse), lm_action,
                f"{progress:.3f}", f"{_state.battery_pct:.1f}",
                f"{elapsed*1000:.1f}"
            ])
            log_f.flush()

            # Dashboard telemetry
            telem = {
                "step": step,
                "timestamp": time.time(),
                "waypoint_idx": wp_idx,
                "waypoint_name": wp_name,
                "total_waypoints": sequencer.total_waypoints,
                "waypoint_names": sequencer.get_all_names(),
                "goal_confidence": round(conf, 4),
                "vx": round(vx, 4),
                "vy": round(vy, 4),
                "repulse": repulse,
                "action": lm_action,
                "progress_pct": round(progress * 100, 1),
                "battery_pct": round(_state.battery_pct, 1),
                "loop_ms": round(elapsed * 1000, 1),
                "control_hz": _state.control_hz,
            }
            _state.telemetry = telem
            write_telemetry(telem)

            # ── E. Actuation ──────────────────────────────────────────────────
            if _state.low_battery:
                print(f"[BATT] Low battery — landing.")
                if not args.dry_run:
                    await controller.land()
                return 'LOW_BATTERY'

            if lm_action == 'LAND':
                print(f"[Flight] ✅ Final landmark '{wp_name}' confirmed. Landing.")
                if not args.dry_run:
                    await controller.land()
                return 'GOAL_REACHED'

            # ADVANCE is already handled inside sequencer.step() — just MOVE
            _state.last_velocity  = {"vx": vx, "vy": vy}
            _state.last_update_ts = time.monotonic()

            if not args.dry_run:
                await controller.move_to({"vx": vx, "vy": vy})

            # ── F. Overrun detection (C1) ─────────────────────────────────────
            if elapsed > loop_budget_s * 1.5:
                _state.consecutive_overruns += 1
                if _state.consecutive_overruns >= OVERRUN_STREAK_MAX:
                    new_hz = max(2, _state.control_hz // 2)
                    print(f"[OVERRUN] Reducing Hz: {_state.control_hz} → {new_hz}")
                    _state.control_hz = new_hz
                    _state.consecutive_overruns = 0
            else:
                _state.consecutive_overruns = 0

            sleep_s = max(0.0, loop_budget_s - elapsed)
            await asyncio.sleep(sleep_s)
            step += 1

    finally:
        log_f.close()
        # Write final telemetry with mission summary
        write_telemetry({**_state.telemetry,
                         "mission_complete": True,
                         "advance_log": sequencer.get_advance_log()})
        print(f"[Log] Flight log saved: {log_path}")

    print(f"[Flight] Max steps ({MAX_STEPS}) reached.")
    return 'MAX_STEPS'


# ── Main ──────────────────────────────────────────────────────────────────────
async def main(args):
    device = get_device()
    print(f"[Init] Device: {device}")

    # 1. Load model
    backbone, visual_enc, goal_enc, depth_enc, memory, path_follower, goal_matcher \
        = load_model(device)

    planner = IntegratedPlanner(
        path_follower, goal_matcher, memory=memory, smoothing=EMA_LAMBDA
    )
    planner.goal_threshold = GOAL_THRESHOLD

    # 2. Load landmark sequence
    sequencer = LandmarkSequencer(
        landmark_dir=args.landmarks,
        goal_encoder=goal_enc,
        device=device,
        preprocess=PREPROCESS,
        waypoint_threshold=WAYPOINT_THRESHOLD,
        confirm_frames=CONFIRM_FRAMES,
        goal_threshold=GOAL_THRESHOLD,
        skip_first_landmark=args.skip_first,
    )

    # 3. Camera + controller
    controller = DroneController(
        system_address=args.address,
        camera_index=args.camera
    )

    if args.dry_run:
        print("\n[DRY-RUN] No drone connected. Commands logged only.\n")
        try:
            await landmark_flight_loop(
                args, controller, planner, sequencer,
                visual_enc, goal_enc, depth_enc, goal_matcher, device
            )
        finally:
            controller.release_camera()
        return

    # 4. Connect + arm + takeoff
    print(f"[Init] Connecting to {args.address} ...")
    await controller.connect()

    print("[Init] Arming ...")
    await controller.drone.action.arm()

    print(f"[Init] Takeoff to {args.altitude}m ...")
    await controller.drone.action.set_takeoff_altitude(args.altitude)
    await controller.drone.action.takeoff()
    await asyncio.sleep(5)

    # 5. Offboard mode
    from mavsdk.offboard import VelocityBodyYawspeed
    await controller.drone.offboard.set_velocity_body_yawspeed(
        VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0)
    )
    await controller.drone.offboard.start()
    print("[Init] Offboard mode active.\n")

    # 6. Safety tasks + flight loop
    _state.last_update_ts = time.monotonic()
    heartbeat  = asyncio.create_task(heartbeat_task(controller, args), name="heartbeat")
    batt_mon   = asyncio.create_task(battery_monitor_task(controller, args), name="battery")

    exit_reason = 'ERROR'
    try:
        exit_reason = await landmark_flight_loop(
            args, controller, planner, sequencer,
            visual_enc, goal_enc, depth_enc, goal_matcher, device
        )
    except KeyboardInterrupt:
        print("\n[INTERRUPT] Ctrl+C — emergency hover.")
    except Exception as e:
        print(f"\n[ERROR] {e}")
    finally:
        _state.stop = True
        for task, name in [(heartbeat, "heartbeat"), (batt_mon, "battery")]:
            try:
                await asyncio.wait_for(task, timeout=1.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                task.cancel()

        await controller.emergency_hover()
        await asyncio.sleep(1)
        if exit_reason != 'GOAL_REACHED':
            await controller.land()
        await controller.drone.offboard.stop()
        controller.release_camera()
        print(f"[Done] Exit: {exit_reason} | Hz: {_state.control_hz}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Landmark-Chain Autonomous Drone Navigation"
    )
    parser.add_argument("--landmarks", required=True,
                        help="Directory containing ordered landmark images")
    parser.add_argument("--address", default="udp://:14540")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--altitude", type=float, default=TAKEOFF_ALTITUDE)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-first", action="store_true",
                        help="Skip landmark[0] (treat it as start reference, not a target)")
    parser.add_argument("--waypoint-threshold", type=float, default=WAYPOINT_THRESHOLD,
                        help=f"Confidence to advance to next waypoint (default: {WAYPOINT_THRESHOLD})")
    parser.add_argument("--goal-threshold", type=float, default=GOAL_THRESHOLD,
                        help=f"Confidence to land at final landmark (default: {GOAL_THRESHOLD})")

    args = parser.parse_args()

    lm_dir = Path(args.landmarks)
    if not lm_dir.exists():
        print(f"[ERROR] Landmark directory not found: '{lm_dir}'")
        sys.exit(1)

    asyncio.run(main(args))
