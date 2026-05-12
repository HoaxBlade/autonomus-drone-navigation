"""
drone_flight.py — Real-World Flight Entry Point
================================================
Wires the trained v2.2 navigation stack to a physical drone via MAVSDK.

Usage:
  # Bench test (no drone needed — logs commands to console)
  python drone_flight.py --goal goal.jpg --dry-run

  # Real indoor flight
  python drone_flight.py --goal goal.jpg --address udp://:14540 --altitude 1.5

  # Real flight with non-default camera
  python drone_flight.py --goal goal.jpg --camera 1 --address udp://:14540
"""

import os
import sys
import asyncio
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from drone_nav.utils.device import get_device
from drone_nav.perception.encoders import (
    PerceptionBackbone, VisualEncoder, GoalEncoder, DepthEncoder, MemoryModule
)
from drone_nav.nav.path_follower import PathFollower
from drone_nav.nav.goal_matcher import GoalMatcher
from drone_nav.control.planner import IntegratedPlanner
from drone_navigator.controller import DroneController

# ── Flight constants (conservative first-flight values) ──────────────────────
WEIGHTS_PATH      = "checkpoints/nav_stack_v2_2.pth"
CONTROL_HZ        = 10          # perception + command rate
MAX_VELOCITY      = 0.5         # m/s — clip planner output for safety
GOAL_THRESHOLD    = 0.85        # confidence to trigger landing
EMA_LAMBDA        = 0.85        # smoother than sim (real vibrations)
MAX_STEPS         = 300         # ~30 s at 10 Hz, then auto-hover
VPR_WINDOW        = 10          # rolling frame buffer for GRU memory
TAKEOFF_ALTITUDE  = 1.5         # metres


# ── Pre-processing (must match training pipeline) ─────────────────────────────
PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def load_model(device):
    """Initialises all network modules and loads checkpoint weights."""
    backbone      = PerceptionBackbone(architecture='resnet18').to(device)
    visual_enc    = VisualEncoder(backbone, use_netvlad=True).to(device)
    goal_enc      = GoalEncoder(backbone).to(device)
    depth_enc     = DepthEncoder(backbone).to(device)
    memory        = MemoryModule(input_dim=visual_enc.output_dim).to(device)
    path_follower = PathFollower(input_dim=memory.hidden_dim).to(device)
    goal_matcher  = GoalMatcher(input_dim=backbone.out_channels).to(device)

    if os.path.exists(WEIGHTS_PATH):
        print(f"[Model] Loading weights from: {WEIGHTS_PATH}")
        ckpt = torch.load(WEIGHTS_PATH, map_location=device, weights_only=True)
        backbone.load_state_dict(ckpt.get('backbone', {}), strict=False)
        visual_enc.load_state_dict(ckpt.get('visual_encoder', {}), strict=False)
        goal_enc.load_state_dict(ckpt.get('goal_encoder', {}), strict=False)
        depth_enc.load_state_dict(ckpt.get('depth_encoder', {}), strict=False)
        path_follower.load_state_dict(ckpt.get('path_follower', {}), strict=False)
        goal_matcher.load_state_dict(ckpt.get('goal_matcher', {}), strict=False)
        if 'memory' in ckpt:
            memory.load_state_dict(ckpt['memory'])
            print("[Model] MemoryModule (GRU) weights loaded.")
        else:
            print("[WARNING] No 'memory' key in checkpoint — GRU using random init.")
    else:
        print(f"[WARNING] No checkpoint at '{WEIGHTS_PATH}'. Running untrained model.")

    # Eval mode — disables dropout, batch-norm uses running stats
    for module in [backbone, visual_enc, goal_enc, depth_enc, memory, path_follower, goal_matcher]:
        module.eval()

    return backbone, visual_enc, goal_enc, depth_enc, memory, path_follower, goal_matcher


def frame_to_tensor(rgb_frame: np.ndarray, device) -> torch.Tensor:
    """Converts an (H, W, 3) RGB numpy array to a (1, 3, 224, 224) tensor."""
    pil_img = Image.fromarray(rgb_frame)
    return PREPROCESS(pil_img).unsqueeze(0).to(device)


def clip_velocity(velocity: list, max_v: float) -> list:
    """Clips each velocity component to [-max_v, max_v]."""
    return [max(-max_v, min(max_v, v)) for v in velocity]


async def flight_loop(args, controller, planner,
                      visual_enc, goal_enc, depth_enc,
                      goal_embedding, device):
    """
    Core 10 Hz perception → planning → actuation loop.
    Returns the exit reason string: 'GOAL_REACHED' | 'MAX_STEPS' | 'ERROR'.
    """
    vpr_memory = []
    step       = 0

    print("\n[Flight] Starting control loop. Press Ctrl+C for emergency hover.")
    print(f"         Max steps: {MAX_STEPS} | Hz: {CONTROL_HZ} | Max speed: {MAX_VELOCITY} m/s\n")

    while step < MAX_STEPS:
        t_start = time.monotonic()

        # ── A. Perception ──────────────────────────────────────────────────
        rgb_frame = await controller.get_camera_frame()
        tensor    = frame_to_tensor(rgb_frame, device)

        with torch.no_grad():
            # Visual place recognition → rolling GRU buffer
            vpr = visual_enc(tensor)
            vpr_memory.append(vpr)
            if len(vpr_memory) > VPR_WINDOW:
                vpr_memory.pop(0)

            # Pad buffer until full
            vpr_seq = torch.stack(vpr_memory, dim=1)           # (1, T, D)
            if vpr_seq.shape[1] < VPR_WINDOW:
                pad     = torch.zeros(1, VPR_WINDOW - vpr_seq.shape[1],
                                      vpr_seq.shape[2], device=device)
                vpr_seq = torch.cat([pad, vpr_seq], dim=1)

            # Goal matching
            obs_v = goal_enc(tensor)                           # (1, 512)

            # Depth for VFH obstacle avoidance
            depth = depth_enc(tensor)                          # (1, 1, 224, 224)

        # ── B. Planning ────────────────────────────────────────────────────
        result   = planner.plan(obs_v, vpr_seq, goal_embedding,
                                depth_map=depth, vpr_obs=vpr)
        action   = result['action']
        velocity = clip_velocity(result['velocity'], MAX_VELOCITY)
        conf     = result.get('goal_confidence', 0.0)
        repulse  = result.get('repulsive_active', False)

        # ── C. Logging ─────────────────────────────────────────────────────
        vx, vy, _ = velocity
        print(
            f"[Step {step:03d}] action={action} | "
            f"vx={vx:+.3f} vy={vy:+.3f} | "
            f"goal_conf={conf:.3f} | "
            f"{'⚠ REPULSE' if repulse else 'clear'}"
        )

        # ── D. Actuation ───────────────────────────────────────────────────
        if action == 'LAND':
            print("[Flight] Goal confidence exceeded threshold. Initiating landing.")
            if not args.dry_run:
                await controller.land()
            return 'GOAL_REACHED'

        if not args.dry_run:
            await controller.move_to({"vx": vx, "vy": vy})

        # ── E. Rate limiting — sleep remainder of 100ms tick ───────────────
        elapsed = time.monotonic() - t_start
        sleep_s = max(0.0, (1.0 / CONTROL_HZ) - elapsed)
        await asyncio.sleep(sleep_s)
        step += 1

    print(f"[Flight] MAX_STEPS ({MAX_STEPS}) reached. Initiating hover/land.")
    return 'MAX_STEPS'


async def main(args):
    device = get_device()
    print(f"[Init] Device: {device}")

    # ── 1. Load model ──────────────────────────────────────────────────────
    backbone, visual_enc, goal_enc, depth_enc, memory, path_follower, goal_matcher \
        = load_model(device)

    planner = IntegratedPlanner(
        path_follower, goal_matcher, memory=memory,
        smoothing=EMA_LAMBDA
    )
    planner.goal_threshold = GOAL_THRESHOLD

    # ── 2. Open camera + encode goal image ────────────────────────────────
    controller = DroneController(
        system_address=args.address,
        camera_index=args.camera
    )
    goal_embedding = controller.load_goal_image(args.goal, goal_enc, device, PREPROCESS)
    print(f"[Init] Goal embedding shape: {goal_embedding.shape}")

    if args.dry_run:
        print("\n[DRY-RUN] Drone NOT connected. Velocity commands will be logged only.\n")
        try:
            await flight_loop(args, controller, planner,
                              visual_enc, goal_enc, depth_enc,
                              goal_embedding, device)
        finally:
            controller.release_camera()
        return

    # ── 3. Connect to flight controller ───────────────────────────────────
    print(f"[Init] Connecting to drone at {args.address} ...")
    await controller.connect()

    # ── 4. Arm + takeoff ──────────────────────────────────────────────────
    print(f"[Init] Arming...")
    await controller.drone.action.arm()

    print(f"[Init] Taking off to {args.altitude}m ...")
    await controller.drone.action.set_takeoff_altitude(args.altitude)
    await controller.drone.action.takeoff()
    await asyncio.sleep(5)          # Wait for stable hover

    # ── 5. Engage offboard mode with an initial zero-velocity setpoint ────
    from mavsdk.offboard import VelocityBodyYawspeed
    await controller.drone.offboard.set_velocity_body_yawspeed(
        VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0)
    )
    await controller.drone.offboard.start()
    print("[Init] Offboard mode active.\n")

    # ── 6. Run control loop ───────────────────────────────────────────────
    exit_reason = 'ERROR'
    try:
        exit_reason = await flight_loop(args, controller, planner,
                                        visual_enc, goal_enc, depth_enc,
                                        goal_embedding, device)
    except KeyboardInterrupt:
        print("\n[INTERRUPT] Ctrl+C received — commanding emergency hover.")
    except Exception as e:
        print(f"\n[ERROR] Unhandled exception: {e}")
        print("[SAFETY] Commanding emergency hover.")
    finally:
        await controller.emergency_hover()
        await asyncio.sleep(1)

        if exit_reason != 'GOAL_REACHED':
            # Land after non-goal exits (timeout, error, interrupt)
            await controller.land()

        await controller.drone.offboard.stop()
        controller.release_camera()
        print(f"[Done] Exit reason: {exit_reason}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Autonomous Drone Navigation — Real-World Flight"
    )
    parser.add_argument(
        "--goal", required=True,
        help="Path to the goal image file (e.g. goal.jpg)"
    )
    parser.add_argument(
        "--address", default="udp://:14540",
        help="MAVSDK system address (default: udp://:14540 for SITL/USB)"
    )
    parser.add_argument(
        "--camera", type=int, default=0,
        help="OpenCV camera index (default: 0 = first USB camera)"
    )
    parser.add_argument(
        "--altitude", type=float, default=TAKEOFF_ALTITUDE,
        help=f"Takeoff altitude in metres (default: {TAKEOFF_ALTITUDE})"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run perception + planning loop without connecting to any drone"
    )

    args = parser.parse_args()

    if not os.path.isfile(args.goal):
        print(f"[ERROR] Goal image not found: '{args.goal}'")
        sys.exit(1)

    asyncio.run(main(args))
