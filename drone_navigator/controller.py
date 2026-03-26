from mavsdk import System
import asyncio
import cv2
import numpy as np

class DroneController:
    def __init__(self, system_address="udp://:14540"):
        self.drone = System()
        self.system_address = system_address
        self.is_connected = False

    async def connect(self):
        print(f"Connecting to drone at {self.system_address}...")
        await self.drone.connect(system_address=self.system_address)
        
        async for state in self.drone.core.connection_state():
            if state.is_connected:
                print(f"-- Connected to drone!")
                self.is_connected = True
                break

    async def get_state(self):
        """
        Return the current state of the drone (position, attitude, battery, etc.).
        """
        # Placeholder for telemetry data
        return {"position": (0, 0, 0)}

    async def get_camera_frame(self):
        """
        Simulate/Get camera frame.
        In simulation, this would be a frame from the simulator's camera.
        """
        # Return a black frame with some noise for now
        return np.zeros((480, 640, 3), dtype=np.uint8)

    async def move_to(self, move_cmd):
        """
        Execute a movement command (velocity or position).
        """
        if not self.is_connected:
            return
            
        vx = move_cmd.get("vx", 0.0)
        vy = move_cmd.get("vy", 0.0)
        vz = move_cmd.get("vz", 0.0)
        
        # Send velocity setpoint
        await self.drone.offboard.set_velocity_body({
            "forward_m_s": vx,
            "right_m_s": vy,
            "down_m_s": -vz, # Note: Z is down in MAVLink
            "yaw_deg": 0.0
        })

    async def land(self):
        print("Landing...")
        await self.drone.action.land()
