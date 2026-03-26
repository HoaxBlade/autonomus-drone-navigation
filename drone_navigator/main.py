import asyncio
from drone_navigator.perception import PerceptionModule
from drone_navigator.planner import PathPlanner
from drone_navigator.controller import DroneController

async def main():
    print("Starting Autonomous Drone Navigator...")
    
    # Initialize modules
    perception = PerceptionModule()
    planner = PathPlanner()
    controller = DroneController()

    # Target Goal (Image path)
    goal_image_path = "target_goal.jpg"
    
    # Main loop
    while True:
        # 1. Get current drone state (from controller)
        current_state = await controller.get_state()
        
        # 2. Get current camera feed and depth (from simulation/interface)
        # For now, let's assume we have a get_frame() function
        current_frame = await controller.get_camera_frame()
        
        # 3. Perception: Detect obstacles and match with goal
        obstacles = perception.detect_obstacles(current_frame)
        goal_reached = perception.match_goal(current_frame, goal_image_path)
        
        if goal_reached:
            print("Goal reached! Landing...")
            await controller.land()
            break
            
        # 4. Planning: Determine next move
        next_move = planner.plan_next_move(current_state, obstacles, goal_image_path)
        
        # 5. Control: Execute move
        await controller.move_to(next_move)
        
        await asyncio.sleep(0.1)

if __name__ == "__main__":
    asyncio.run(main())
