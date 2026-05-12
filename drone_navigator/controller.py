from mavsdk import System
from mavsdk.offboard import VelocityBodyYawspeed, OffboardError
import asyncio
import cv2
import numpy as np
from PIL import Image
import torch

class DroneController:
    def __init__(self, system_address="udp://:14540", camera_index=0):
        self.drone = System()
        self.system_address = system_address
        self.is_connected = False

        # A2 — Live camera
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera at index {camera_index}. "
                "Check USB connection and set correct camera_index."
            )

    async def connect(self):
        print(f"Connecting to drone at {self.system_address}...")
        await self.drone.connect(system_address=self.system_address)
        
        async for state in self.drone.core.connection_state():
            if state.is_connected:
                print(f"-- Connected to drone!")
                self.is_connected = True
                break

    async def get_state(self) -> dict:
        """
        Returns live FC telemetry: position, velocity, heading.
        Each field is read from a single-shot async generator.
        """
        state: dict = {}
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

    async def get_battery(self) -> float:
        """
        C2 — Returns the current battery charge as a percentage (0.0 – 100.0).
        Reads a single value from the MAVSDK battery telemetry stream.
        Returns -1.0 if telemetry is unavailable (not connected).
        """
        try:
            async for battery in self.drone.telemetry.battery():
                return battery.remaining_percent * 100.0
        except Exception:
            return -1.0

    async def get_camera_frame(self):
        """
        Reads the latest frame from the onboard USB/CSI camera.
        Returns an (H, W, 3) RGB numpy array.
        """
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError(
                "Camera read failed. Check that the camera is still connected."
            )
        # OpenCV reads BGR — convert to RGB for the perception stack
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def load_goal_image(self, goal_image_path: str, goal_encoder, device, preprocess):
        """
        A4 — Loads and pre-encodes the goal image ONCE at startup.
        Keeps the (1, 512) embedding tensor in memory; avoids re-encoding every step.

        Args:
            goal_image_path: Path to the goal JPEG/PNG file.
            goal_encoder:    GoalEncoder module (already on device).
            device:          torch.device.
            preprocess:      torchvision transforms pipeline (Resize+ToTensor+Normalize).

        Returns:
            goal_embedding: Tensor of shape (1, backbone.out_channels).
        """
        img = Image.open(goal_image_path).convert("RGB")
        tensor = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            goal_embedding = goal_encoder(tensor)
        print(f"[GoalEncoder] Goal image encoded from: {goal_image_path}")
        return goal_embedding

    def release_camera(self):
        """Releases the camera resource. Call on shutdown."""
        if self.cap.isOpened():
            self.cap.release()

    async def move_to(self, move_cmd):
        """
        A1 — Send a body-frame velocity setpoint to the flight controller.

        vz is intentionally locked to 0: the Pixhawk's barometer/altitude-hold
        mode manages Z until SLAM is integrated (roadmap Phase 5).
        """
        if not self.is_connected:
            return

        vx = float(move_cmd.get("vx", 0.0))
        vy = float(move_cmd.get("vy", 0.0))
        # vz deliberately ignored — altitude hold via FC

        await self.drone.offboard.set_velocity_body_yawspeed(
            VelocityBodyYawspeed(
                forward_m_s=vx,
                right_m_s=vy,
                down_m_s=0.0,       # locked: FC handles altitude
                yawspeed_deg_s=0.0  # no yaw commands yet
            )
        )

    async def emergency_hover(self):
        """Immediately commands zero velocity. Call in exception handlers."""
        if self.is_connected:
            await self.drone.offboard.set_velocity_body_yawspeed(
                VelocityBodyYawspeed(
                    forward_m_s=0.0,
                    right_m_s=0.0,
                    down_m_s=0.0,
                    yawspeed_deg_s=0.0
                )
            )

    async def land(self):
        print("Landing...")
        await self.drone.action.land()
