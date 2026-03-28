# Autonomous Drone Navigation

This project aims to provide autonomous navigation for drones using a target image as a goal.

## Project Structure
- `drone_navigator/`: Core system components.
  - `perception.py`: Computer vision logic for obstacle detection and goal recognition.
  - `planner.py`: Path planning algorithms.
  - `controller.py`: Drone interface (MAVSDK/PX4).
  - `main.py`: Entry point.

## How to Run (Simulation)
1. Install dependencies: `pip install -r requirements.txt`
2. Run PX4 SITL (Software In The Loop) or a compatible simulator.
3. Run the navigator: `python drone_navigator/main.py `
