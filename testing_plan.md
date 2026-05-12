# Testing Plan — Autonomous Drone Navigation v2.2

**System under test:** `drone_flight.py` + `drone_navigator/` + `drone_nav/`
**Trained weights:** `checkpoints/nav_stack_v2_2.pth`
**Last benchmark:** SR=100%, ESS=1.00, Deviation=0.035 m/s (TartanAir Easy P001)

---

## Test Stages Overview

```
Stage 0 — Software Unit Tests       (no hardware)
Stage 1 — Dry-Run Loop Test         (no hardware)
Stage 2 — Hardware-in-the-Loop      (drone on bench, props OFF)
Stage 3 — Tethered Indoor Flight    (props ON, safety tether)
Stage 4 — Free Indoor Flight        (short range, safety net)
```

Each stage must fully **PASS** before proceeding to the next.

---

## Stage 0 — Software Unit Tests
> **Hardware required:** None  
> **Time estimate:** 30 min

### S0-1 · Camera Opens
```bash
python -c "
import cv2
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
print('PASS' if ret and frame is not None else 'FAIL')
print(f'Frame shape: {frame.shape}')
cap.release()
"
```
**Pass:** Prints `PASS` and a valid shape like `(480, 640, 3)`  
**Fail action:** Try `camera_index=1`, check USB cable, try `cv2.VideoCapture(1)`

---

### S0-2 · Model Forward Pass
```bash
python drone_navigator/main.py
```
**Pass:** Prints `Perception Status: Success`, depth/VPR shapes, and a velocity without errors  
**Fail action:** Check PyTorch install, verify `checkpoints/nav_stack_v2_2.pth` exists

---

### S0-3 · Checkpoint Integrity
```bash
python -c "
import torch, os
path = 'checkpoints/nav_stack_v2_2.pth'
assert os.path.exists(path), 'Checkpoint not found'
ckpt = torch.load(path, map_location='cpu', weights_only=True)
required = ['backbone','visual_encoder','goal_encoder','depth_encoder',
            'path_follower','goal_matcher','memory']
missing = [k for k in required if k not in ckpt]
print('PASS' if not missing else f'FAIL — missing keys: {missing}')
"
```
**Pass:** Prints `PASS`  
**Fail action:** Re-run `python train/trainer.py` to regenerate checkpoint; check all keys are saved in `get_checkpoint()`

---

### S0-4 · Goal Image Encoding
Prepare a test goal image first:
```bash
# Capture one frame from camera as the goal image
python -c "
import cv2
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cv2.imwrite('goal.jpg', frame)
cap.release()
print('goal.jpg saved')
"
```

Then verify goal encoding:
```bash
python -c "
import torch
from PIL import Image
from torchvision import transforms
from drone_nav.perception.encoders import PerceptionBackbone, GoalEncoder

device = 'cpu'
bb  = PerceptionBackbone('resnet18').to(device)
enc = GoalEncoder(bb).to(device)

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

img = Image.open('goal.jpg').convert('RGB')
t   = preprocess(img).unsqueeze(0).to(device)
with torch.no_grad():
    emb = enc(t)
print('PASS' if emb.shape == (1, 512) else f'FAIL — shape {emb.shape}')
"
```
**Pass:** Prints `PASS`

---

### S0-5 · GRU Memory Loaded
```bash
python -c "
import torch
from drone_nav.perception.encoders import PerceptionBackbone, VisualEncoder, MemoryModule

bb  = PerceptionBackbone('resnet18')
ve  = VisualEncoder(bb)
mem = MemoryModule(input_dim=ve.output_dim)

ckpt = torch.load('checkpoints/nav_stack_v2_2.pth', map_location='cpu', weights_only=True)
if 'memory' in ckpt:
    mem.load_state_dict(ckpt['memory'])
    print('PASS — GRU weights loaded')
else:
    print('WARN — no memory key; GRU will use random init')
"
```
**Pass:** Prints `PASS — GRU weights loaded`

---

## Stage 1 — Dry-Run Loop Test
> **Hardware required:** USB camera only  
> **Time estimate:** 15 min

### S1-1 · Dry-Run Launches
```bash
python drone_flight.py --goal goal.jpg --dry-run
```
**Pass criteria (all must hold for ≥ 60 steps):**

| Check | Expected |
|---|---|
| No Python exceptions | ✅ |
| `action` field is `MOVE` or `LAND` | ✅ |
| `vx`, `vy` values within `[-0.5, 0.5]` | ✅ |
| `goal_confidence` printed each step | ✅ |
| Process exits cleanly on Ctrl+C | ✅ |

---

### S1-2 · Velocity Sanity — Open Scene
Point the camera at an empty wall/ceiling (not the goal image).

**Expected:** `goal_confidence` stays **below 0.5**, `action=MOVE`, non-zero velocities  
**Fail:** `goal_confidence > 0.85` on a blank scene → GoalMatcher weights corrupt or goal image was taken at the same spot

---

### S1-3 · Goal Detection Test
Point the camera **directly at `goal.jpg`** (printed or on a screen).

**Expected:** `goal_confidence` rises above `0.85` within ~5 steps, `action=LAND` triggered  
**Fail action:** Lower `GOAL_THRESHOLD` in `drone_flight.py` from 0.85 → 0.75; re-test

---

### S1-4 · VFH Repulsion Test
Hold an object 20–30 cm in front of the camera.

**Expected:** `⚠ REPULSE` appears in console output  
**Fail action:** Check `depth_encoder` forward pass; verify `DepthEncoder` is in `eval()` mode

---

### S1-5 · Emergency Hover on Ctrl+C
Start dry-run, press **Ctrl+C** after ~10 steps.

**Expected:** Prints `[INTERRUPT] Ctrl+C received — commanding emergency hover.`, exits cleanly, camera released  
**Fail action:** Check `finally` block in `flight_loop` and `main`

---

## Stage 2 — Hardware-in-the-Loop (Props OFF)
> **Hardware required:** Jetson/PC + camera + Pixhawk (USB) + QGroundControl or MAVLink Inspector  
> **Drone state:** Powered, props REMOVED, flat surface  
> **Time estimate:** 1–2 hours

### S2-1 · MAVSDK Connection
```bash
python -c "
import asyncio
from mavsdk import System

async def test():
    drone = System()
    await drone.connect(system_address='udp://:14540')
    async for state in drone.core.connection_state():
        print('PASS — Connected!' if state.is_connected else 'waiting...')
        if state.is_connected:
            break

asyncio.run(test())
"
```
**Pass:** Prints `PASS — Connected!`  
**Fail action:** Check USB cable, verify PX4 SITL or hardware is on `udp://:14540`

---

### S2-2 · Arm + Offboard Mode Test
```bash
python drone_flight.py --goal goal.jpg --address udp://:14540 --dry-run
```
> ⚠️ This still uses `--dry-run` — it WILL connect but will NOT send velocity commands.

**Pass:** `[Init] Connecting...`, `-- Connected to drone!`, loop starts printing steps  
**Fail action:** Check flight controller mode allows offboard; ensure RC transmitter is off or in offboard-enable mode

---

### S2-3 · Velocity Command Inspection
Remove `--dry-run` flag. Monitor in **QGroundControl → MAVLink Inspector → SET_ATTITUDE_TARGET / SET_POSITION_TARGET_LOCAL_NED**.

```bash
python drone_flight.py --goal goal.jpg --address udp://:14540
```
**Pass criteria:**

| Check | Expected |
|---|---|
| Commands arriving at ~10 Hz | ✅ |
| `vx` range within `[-0.5, 0.5]` m/s | ✅ |
| `vy` range within `[-0.5, 0.5]` m/s | ✅ |
| `vz = 0.0` always | ✅ |
| `yawspeed = 0.0` always | ✅ |
| LAND command stops the command stream | ✅ |

**Fail action:** Check `VelocityBodyYawspeed` import in `controller.py`; verify `move_to()` is awaited correctly

---

### S2-4 · Emergency Hover Triggers (Simulated Fault)
While running S2-3, disconnect the camera USB cable.

**Expected:** `[ERROR] Camera read failed.` → `emergency_hover()` → `land()` called  
**Fail action:** Verify `RuntimeError` propagates out of `get_camera_frame()` and is caught in `main()`

---

## Stage 3 — Tethered Indoor Flight
> **Hardware required:** Full drone, indoor space ≥ 4×4m, ≥ 2m ceiling, **safety tether**  
> **Personnel:** Pilot (safety override on RC) + observer  
> **Props:** ON  
> **Altitude:** 1.0m (reduced from default for first tethered test)  
> **Tether length:** ≤ 1.5m to limit lateral drift

### Pre-flight Checklist
- [ ] Stage 0 all PASS
- [ ] Stage 1 all PASS
- [ ] Stage 2 all PASS
- [ ] Tether attached to drone frame (not props)
- [ ] RC transmitter on, pilot has thumb on override switch
- [ ] `goal.jpg` placed/printed at the far end of the room
- [ ] Observer watching drone attitude and motor response

---

### S3-1 · Hover Stability Test
```bash
python drone_flight.py --goal goal.jpg --address udp://:14540 --altitude 1.0
```
Cover the camera with your hand (no goal visible) for 30 seconds.

**Pass criteria:**

| Check | Expected |
|---|---|
| Drone holds altitude ±0.2m | ✅ |
| No aggressive lateral movement | ✅ |
| EMA smoothing visible (no jitter) | ✅ |
| No safety interventions by pilot | ✅ |

**Fail action:** Increase `EMA_LAMBDA` to 0.90, reduce `MAX_VELOCITY` to 0.3

---

### S3-2 · Goal-Directed Movement
Uncover camera and face drone toward `goal.jpg`.

**Pass criteria:**

| Check | Expected |
|---|---|
| Drone moves toward goal direction | ✅ |
| `goal_confidence` increases as drone nears goal | ✅ |
| Drone slows as confidence exceeds 0.6 (deceleration logic) | ✅ |
| Lands when confidence > 0.85 | ✅ |
| No collision with walls/objects | ✅ |

---

### S3-3 · Obstacle Avoidance (Hand Test)
During S3-2, slowly move a hand in front of the camera at ~30cm.

**Pass:** Drone lateral velocity shifts away from hand (VFH repulsion), console shows `⚠ REPULSE`  
**Fail action:** Adjust VFH `safety_limit` threshold in `planner.py` from 0.25 → 0.35

---

## Stage 4 — Free Indoor Flight
> **Hardware required:** Same as Stage 3, no tether  
> **Space:** ≥ 6×6m, ≥ 2.5m ceiling, soft crash net recommended  
> **Personnel:** Safety pilot + observer + logger  
> **Altitude:** 1.5m (default)

### Pre-flight Checklist
- [ ] Stage 3 all PASS, at least 3 consecutive successful runs
- [ ] Crash net set up around flight area
- [ ] Battery fully charged (≥ 95%)
- [ ] `goal.jpg` placed at far end (≥ 4m from start)
- [ ] Pilot briefed on override procedure

---

### S4-1 · Full Mission Run
```bash
python drone_flight.py --goal goal.jpg --address udp://:14540
```

**Record the following for each run:**

| Metric | Run 1 | Run 2 | Run 3 | Target |
|---|---|---|---|---|
| Success (reached goal) | | | | ≥ 2/3 |
| Steps to goal | | | | ≤ 150 |
| Collisions | | | | 0 |
| Safety pilot interventions | | | | 0 |
| Goal confidence at landing | | | | ≥ 0.85 |
| Max observed speed (m/s) | | | | ≤ 0.5 |

---

### S4-2 · Varied Starting Positions
Run 3 missions starting from different positions/orientations relative to the goal.

**Pass:** ≥ 2/3 missions reach goal without pilot intervention  
**Fail action:** Review GRU memory buffer — check if VPR embeddings are too similar across positions (NetVLAD not discriminative enough for this environment)

---

### S4-3 · Failure Recovery
Deliberately obscure the goal midway through a mission for 5 seconds.

**Expected:** Drone slows or hovers (`goal_confidence` drops), resumes movement when goal is visible again  
**Pass:** No crash, no erratic behavior

---

## Pass/Fail Summary Table

| Stage | Test | Status |
|---|---|---|
| S0-1 | Camera opens | ⬜ |
| S0-2 | Model forward pass | ⬜ |
| S0-3 | Checkpoint integrity | ⬜ |
| S0-4 | Goal image encoding | ⬜ |
| S0-5 | GRU weights loaded | ⬜ |
| S1-1 | Dry-run launches | ⬜ |
| S1-2 | Velocity sanity — open scene | ⬜ |
| S1-3 | Goal detection test | ⬜ |
| S1-4 | VFH repulsion test | ⬜ |
| S1-5 | Emergency hover on Ctrl+C | ⬜ |
| S2-1 | MAVSDK connection | ⬜ |
| S2-2 | Arm + offboard mode | ⬜ |
| S2-3 | Velocity command inspection | ⬜ |
| S2-4 | Emergency hover on fault | ⬜ |
| S3-1 | Hover stability | ⬜ |
| S3-2 | Goal-directed movement | ⬜ |
| S3-3 | Obstacle avoidance (hand) | ⬜ |
| S4-1 | Full mission run (3×) | ⬜ |
| S4-2 | Varied starting positions | ⬜ |
| S4-3 | Failure recovery | ⬜ |

> Update status: ⬜ Not run · ✅ Pass · ❌ Fail · ⚠️ Pass with notes

---

## Known Limitations at This Stage

| Limitation | Implication | Workaround |
|---|---|---|
| Model trained on TartanAir sim only | May drift in real scenes | Fine-tune on real indoor images |
| No absolute position tracking | Drift on long paths (>10m) | Restrict to ≤ 5m missions |
| Altitude via FC barometer only | Wind or ground effect causes ±0.3m | Fly in calm indoor space |
| Goal must be visually similar to training views | Extreme viewpoint changes → low confidence | Take goal photo from the same height as flight altitude |
| No hardware kill-switch in software | Requires pilot RC override | Always fly with safety pilot |

---

## Quick Reference Commands

```bash
# Stage 0 — All software tests
python drone_navigator/main.py

# Stage 1 — Dry-run
python drone_flight.py --goal goal.jpg --dry-run

# Stage 2 — Connected, dry-run (no velocity sent)
python drone_flight.py --goal goal.jpg --address udp://:14540 --dry-run

# Stage 3/4 — Real flight (1.5m altitude)
python drone_flight.py --goal goal.jpg --address udp://:14540 --altitude 1.5

# Real flight with non-default camera
python drone_flight.py --goal goal.jpg --camera 1 --address udp://:14540

# Post-flight evaluation on TartanAir dataset
python evaluate_metrics.py
```
