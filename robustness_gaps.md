# Robustness Gap Analysis — Autonomous Drone Navigation v2.2

All gaps are analyzed from the actual code. Organized by risk level.

---

## 🔴 CRITICAL — Can cause a crash or flyaway

### C1 — No Watchdog Timer on the Control Loop [x]
**File:** `drone_flight.py` — `flight_loop()`  
**Problem:** If PyTorch inference takes >100ms (CPU spike, thermal throttle on Jetson), the loop runs slower than 10Hz but the drone doesn't know — it just waits. Offboard mode on PX4 exits if setpoints stop arriving for >500ms, then the drone enters failsafe (usually: land or hold).  
**Risk:** Unpredictable FC behavior if Jetson stalls.

**Fix:**
```python
# After every move_to(), also schedule a "heartbeat" setpoint in case
# the next loop iteration is late:
OFFBOARD_TIMEOUT = 0.4  # PX4 exits offboard after 500ms

# In flight_loop, add a deadline check:
elapsed = time.monotonic() - t_start
if elapsed > (1.0 / CONTROL_HZ) * 1.5:
    print(f"[WARN] Loop overran: {elapsed*1000:.0f}ms (budget: {1000/CONTROL_HZ:.0f}ms)")
```

---

### C2 — No Battery Level Monitoring [x]
**File:** `drone_flight.py` / `controller.py`  
**Problem:** `get_state()` returns `{"position": (0,0,0)}` — a placeholder. No battery telemetry is read. If the battery dies mid-flight, the FC handles it, but the software has no warning to initiate a controlled landing early.

**Fix — add to `controller.py`:**
```python
async def get_battery(self) -> float:
    """Returns remaining battery percentage (0.0–100.0)."""
    async for battery in self.drone.telemetry.battery():
        return battery.remaining_percent * 100.0

# In flight_loop, check every 30 steps:
if step % 30 == 0 and not args.dry_run:
    battery = await controller.get_battery()
    if battery < 20.0:
        print(f"[SAFETY] Battery critical: {battery:.1f}%. Returning to land.")
        return 'LOW_BATTERY'
```

---

### C3 — LAND Triggered by a Single Noisy Frame
**File:** `drone_nav/control/planner.py` — line 52  
**Problem:** `if goal_confidence > self.goal_threshold` triggers landing on ONE frame above 0.85. A single glare, lens flare, or similar-looking scene patch can trigger an early landing.

**Fix — add a confirmation window:**
```python
# In IntegratedPlanner.__init__:
self.goal_confidence_history = []
self.goal_confirm_window = 5   # must be high for N consecutive frames

# In plan(), replace single-frame check:
self.goal_confidence_history.append(goal_confidence)
if len(self.goal_confidence_history) > self.goal_confirm_window:
    self.goal_confidence_history.pop(0)

# Only land if ALL recent frames are confident
if (len(self.goal_confidence_history) == self.goal_confirm_window and
        all(c > self.goal_threshold for c in self.goal_confidence_history)):
    ...  # LAND
```

---

### C4 — No Geofencing / Boundary Enforcement
**File:** `drone_flight.py`  
**Problem:** The drone has no concept of spatial boundaries. If the model predicts incorrect velocities, the drone can fly indefinitely in any direction.

**Fix:**
```python
# Add to flight_loop — read position from FC telemetry
# and compare against a max displacement from takeoff point:
MAX_DISPLACEMENT = 5.0  # metres (for indoor testing)

async for pos in controller.drone.telemetry.position():
    current_pos = (pos.latitude_deg, pos.longitude_deg, pos.absolute_altitude_m)
    break

# Compare against recorded home position — if exceeded, hover+land
```
> Note: Requires implementing real telemetry in `get_state()` first (see M1).

---

## 🟠 HIGH — Degrades mission reliability significantly

### H1 — No Telemetry Logging
**File:** `drone_flight.py`  
**Problem:** Every flight step is printed to console but nothing is persisted. If the drone crashes, there is no post-mortem data.

**Fix — add CSV logging:**
```python
import csv
log_path = f"logs/flight_{int(time.time())}.csv"
os.makedirs("logs", exist_ok=True)

with open(log_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['step','timestamp','vx','vy','goal_conf','repulse','action'])
    # In flight_loop:
    writer.writerow([step, time.time(), vx, vy, conf, repulse, action])
```

---

### H2 — VFH Only Has 3 Sectors (Very Coarse)
**File:** `drone_nav/control/planner.py` — lines 63–85  
**Problem:** The depth map is divided into only Left/Center/Right. A narrow obstacle (pole, door frame) in between sectors can be missed entirely.

**Fix — increase to 5 sectors:**
```python
# 5 sectors: Far-Left, Left, Center, Right, Far-Right
sectors = torch.chunk(depth_map, 5, dim=-1)  # splits width into 5
sector_means = [torch.mean(s).item() for s in sectors]

# Repulsive vectors for each sector pair
# Far-Left obstacle → push strongly right
# Left obstacle → push moderately right
# etc.
```

---

### H3 — No Pre-flight Confidence Sanity Check
**File:** `drone_flight.py` — `main()`  
**Problem:** The drone arms and takes off without verifying that the goal image actually produces a distinguishable embedding from the current scene. If the goal image and the takeoff position look similar, the drone will immediately land after takeoff.

**Fix — add a pre-flight check before arm:**
```python
async def preflight_goal_sanity_check(controller, goal_enc, goal_embedding, device):
    """Checks that goal_confidence at takeoff position is LOW (< 0.4)."""
    frame = await controller.get_camera_frame()
    tensor = frame_to_tensor(frame, device)
    with torch.no_grad():
        obs_v = goal_enc(tensor)
    from drone_nav.nav.goal_matcher import GoalMatcher
    conf = goal_matcher(obs_v, goal_embedding).item()
    if conf > 0.5:
        raise RuntimeError(
            f"Pre-flight FAIL: goal_confidence={conf:.2f} at takeoff position. "
            "The goal image looks too similar to the start position. "
            "Use a more distinctive goal image or move the drone further from goal."
        )
    print(f"[Pre-flight] Goal sanity check PASS — confidence at start: {conf:.3f}")
```

---

### H4 — `get_state()` Returns a Placeholder
**File:** `drone_navigator/controller.py` — line 37  
**Problem:** `return {"position": (0, 0, 0)}` — always. No real telemetry. Code that depends on position data (geofencing, logging) cannot work.

**Fix:**
```python
async def get_state(self) -> dict:
    """Returns real FC telemetry: position, velocity, battery, heading."""
    state = {}
    async for pos in self.drone.telemetry.position():
        state['lat']      = pos.latitude_deg
        state['lon']      = pos.longitude_deg
        state['altitude'] = pos.absolute_altitude_m
        break
    async for vel in self.drone.telemetry.velocity_ned():
        state['vx'] = vel.north_m_s
        state['vy'] = vel.east_m_s
        state['vz'] = vel.down_m_s
        break
    return state
```

---

### H5 — No RC Signal Loss / Failsafe Awareness
**File:** `drone_flight.py`  
**Problem:** If the RC transmitter loses connection mid-flight, the FC may enter failsafe independently. The Python stack doesn't know this happened and keeps sending velocity commands into the void.

**Fix:**
```python
# Monitor RC status in parallel
async def monitor_rc(controller, stop_event):
    async for rc in controller.drone.telemetry.rc_status():
        if not rc.is_available:
            print("[SAFETY] RC signal lost — triggering emergency hover.")
            stop_event.set()
            break

# In main(), run as parallel task:
stop_event = asyncio.Event()
asyncio.create_task(monitor_rc(controller, stop_event))
# In flight_loop, check stop_event each step
```

---

## 🟡 MEDIUM — Reduces reliability, manageable for indoor testing

### M1 — No Loop Timing Metrics
**File:** `drone_flight.py` — `flight_loop()`  
**Problem:** Loop overruns are silently swallowed by `max(0.0, sleep_s)`. You can't tell if inference is keeping up with 10Hz without measuring it.

**Fix:** Log actual loop time each step. If consistently >90ms on CPU, consider reducing to 5Hz or switching to a lighter backbone.

---

### M2 — Confidence Threshold is Static
**File:** `drone_flight.py` — `GOAL_THRESHOLD = 0.85`  
**Problem:** The right threshold depends on the environment. Bright outdoor scenes produce different confidence distributions than dark indoor ones. A single hardcoded value will be wrong in some conditions.

**Fix — adaptive threshold based on confidence history:**
```python
# Track the rolling max confidence seen so far
# Only land when: current_conf > 0.85 AND current_conf > (rolling_max * 0.95)
# This prevents premature landing on a false peak
```

---

### M3 — No Inference Fallback on Perception Failure
**File:** `drone_flight.py` — `flight_loop()`  
**Problem:** If `depth_enc(tensor)` or `visual_enc(tensor)` throws (OOM, NaN in input), the entire loop crashes and the emergency hover activates. Better to issue a hover command that step and retry next frame.

**Fix — wrap inference in try/except per-step:**
```python
try:
    with torch.no_grad():
        vpr  = visual_enc(tensor)
        obs_v = goal_enc(tensor)
        depth = depth_enc(tensor)
except Exception as e:
    print(f"[WARN] Inference error step {step}: {e}. Hovering this step.")
    if not args.dry_run:
        await controller.emergency_hover()
    step += 1
    continue
```

---

### M4 — Sim-to-Real Gap (Model Only Trained on TartanAir)
**File:** `train/trainer.py`  
**Problem:** The model has never seen a real camera image. Indoor scenes, lens distortion, motion blur, and different lighting will all degrade perception quality. The current augmentation (ColorJitter + GaussianNoise) only partially bridges this gap.

**Mitigations:**
1. Collect 1–2 hours of video walking through the flight environment → use as fine-tuning data
2. Add motion blur augmentation to the training pipeline
3. Add random brightness/gamma augmentation to the training pipeline
4. Consider using MiDaS pretrained depth instead of the learned `DepthEncoder`

---

### M5 — No NaN/Inf Guard on Velocity Output
**File:** `drone_flight.py` — `clip_velocity()`  
**Problem:** If the model outputs `NaN` or `Inf` (can happen with random init or corrupted input), `clip_velocity()` silently passes `NaN` to MAVSDK, which sends garbage to the FC.

**Fix:**
```python
def clip_velocity(velocity: list, max_v: float) -> list:
    result = []
    for v in velocity:
        if not isinstance(v, (int, float)) or v != v:  # NaN check
            print(f"[WARN] NaN/Inf in velocity output: {velocity}. Substituting 0.")
            return [0.0, 0.0, 0.0]
        result.append(max(-max_v, min(max_v, v)))
    return result
```

---

## 🟢 LOW — Polish / operational quality

### L1 — No Structured Log File
Console output only. Add JSON/CSV logging per flight with timestamp, all step data, and exit reason. Essential for debugging real flight behavior post-hoc.

### L2 — No Altitude Monitoring in Software
The FC holds altitude but the software never reads it. Add a check: if `abs(current_alt - target_alt) > 0.5m`, log a warning.

### L3 — No Goal Image Blur/Quality Check
Before encoding the goal image, check that it's not blurry or underexposed. A blurry goal image → always low confidence → drone never lands.
```python
laplacian_var = cv2.Laplacian(cv2.cvtColor(goal_img, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var()
if laplacian_var < 50:
    print(f"[WARN] Goal image may be blurry (sharpness={laplacian_var:.1f}). Consider retaking.")
```

### L4 — No Version Check on Checkpoint
If you retrain and accidentally load an old incompatible checkpoint, `strict=False` silently accepts it. Add a version field to the checkpoint dict and verify it on load.

---

## Summary Priority Table

| ID | Severity | Effort | Fix Now? |
|---|---|---|---|
| C1 — No watchdog timer | 🔴 Critical | Low | ✅ Yes |
| C2 — No battery monitor | 🔴 Critical | Low | ✅ Yes |
| C3 — Single-frame LAND | 🔴 Critical | Low | ✅ Yes |
| C4 — No geofencing | 🔴 Critical | Medium | ✅ Yes (before first flight) |
| H1 — No telemetry log | 🟠 High | Low | ✅ Yes |
| H2 — 3-sector VFH only | 🟠 High | Low | ✅ Yes |
| H3 — No pre-flight check | 🟠 High | Low | ✅ Yes |
| H4 — Fake `get_state()` | 🟠 High | Medium | ✅ Yes |
| H5 — No RC loss detection | 🟠 High | Medium | Before outdoor flight |
| M1 — No loop timing | 🟡 Medium | Low | Optional |
| M2 — Static threshold | 🟡 Medium | Low | Optional |
| M3 — No per-step fallback | 🟡 Medium | Low | Optional |
| M4 — Sim-to-real gap | 🟡 Medium | High | Fine-tune on real data |
| M5 — NaN guard | 🟡 Medium | Low | ✅ Yes |
| L1–L4 | 🟢 Low | Low | Before outdoor flight |

**Implement C1–C4 + H1–H3 + M5 before any real flight.** That's 8 fixes, all straightforward.
