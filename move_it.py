import asyncio
from mavsdk import System
from mavsdk.offboard import OffboardError, PositionNedYaw, VelocityNedYaw

async def run():
    drone = System()
    # Connect to the drone (companion computer link or SITL simulation link)
    await drone.connect(system_address="serial:///dev/ttyS6:921600")

    print("Waiting for drone to connect...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("Drone connected!")
            break

    # OPTICAL FLOW HEALTH CHECK
    print("Waiting for Optical Flow / Local Position Lock...")
    async for health in drone.telemetry.health():
        if health.is_local_position_ok:
            print("✔ Optical Flow initialized! Local position estimate is healthy.")
            break

    # 1. SET THE SYSTEM TAKEOFF HEIGHT PARAMETER
    # Configures the default autopilot action altitude to exactly 2.0 meters
    print("-- Setting default takeoff height parameter to 2.0 meters...")
    await drone.action.set_takeoff_altitude(2.0)
    
    print("-- Setting local arming point as Home reference...")
    
    print("-- Arming Motors")
    await drone.action.arm()

    # PRE-STREAM INITIAL SETPOINT (Mandatory for Offboard)
    # Zero movement constraint to safely bridge into offboard mode
    initial_pos = PositionNedYaw(0.0, 0.0, 0.0, 0.0)
    initial_vel = VelocityNedYaw(0.0, 0.0, 0.0, 0.0)
    await drone.offboard.set_position_velocity_ned(initial_pos, initial_vel)

    print("-- Engaging Offboard Mode")
    try:
        await drone.offboard.start()
    except OffboardError as error:
        print(f"Offboard failed: {error._result.result}")
        await drone.action.disarm()
        return

    # 2. OFFBOARD TAKEOFF TO 2 METERS
    # Command: Rise 2 meters straight up (Down is negative, so -2.0)
    # Constraint: Restricting ascent speed to 0.5 m/s
    print("-- Taking off: Ascending to 2m using Optical Flow + Rangefinder")
    target_pos = PositionNedYaw(0.0, 0.0, -2.0, 0.0)
    target_vel = VelocityNedYaw(0.0, 0.0, -0.5, 0.0)
    await drone.offboard.set_position_velocity_ned(target_pos, target_vel)
    await asyncio.sleep(6)

    # Command: Move 3 meters forward (North) at a restricted 1.0 m/s velocity constraint
    print("-- Flying 3m forward inside local frame")
    target_pos = PositionNedYaw(3.0, 0.0, -2.0, 0.0)
    target_vel = VelocityNedYaw(1.0, 0.0, 0.0, 0.0)
    await drone.offboard.set_position_velocity_ned(target_pos, target_vel)
    await asyncio.sleep(6)

    # SAFE EXIT
    print("-- Stopping Offboard & Initiating Land")
    try:
        await drone.offboard.stop()
    except OffboardError as error:
        print(f"Stop failed: {error._result.result}")

    # Uses the rangefinder to descend cleanly to the ground
    await drone.action.land()

if __name__ == "__main__":
    asyncio.run(run())
