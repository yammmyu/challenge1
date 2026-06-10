"""
Example: how a teammate's flight code integrates the perception module.

Camera ownership: the FLIGHT side owns the RealSense camera (same pipe as the
video stream) and the drone pose. It grabs frames and hands them to perception.
The only contract is:

    result = perc.process_frame(color, depth, pose={'north', 'east', 'yaw'})

Run this as a reference, or copy the marked sections into the real mission.
"""

import asyncio

from mavsdk import System

from perception_module import PerceptionModule
from realsense_manager import RealSenseManager


async def position_monitor(drone, state):
    """Keep the latest NED position + yaw available for the perception calls."""
    async def _pos():
        async for pv in drone.telemetry.position_velocity_ned():
            if state['stop']:
                break
            state['pos'] = pv.position

    async def _yaw():
        async for att in drone.telemetry.attitude_euler():
            if state['stop']:
                break
            state['yaw'] = att.yaw_deg

    await asyncio.gather(_pos(), _yaw())


async def perception_loop(cam, perc, state):
    """Runs the whole mission, in parallel with flight control.

    The flight side grabs a frame, reads the current pose, and asks perception
    to process it. grab() + YOLO are blocking, so run them in a worker thread to
    keep the flight event loop responsive.
    """
    def grab_and_process(pose):
        # Runs in a worker thread.
        try:
            ok = cam.grab()
        except RuntimeError:
            ok = False
        if not ok:
            return None
        return perc.process_frame(cam.get_color(), cam.get_depth(), pose)

    while not state['stop']:
        pos = state.get('pos')
        pose = None
        if pos is not None:
            pose = {'north': pos.north_m, 'east': pos.east_m, 'yaw': state['yaw']}

        result = await asyncio.to_thread(grab_and_process, pose)
        if result is None:
            await asyncio.sleep(0.02)        # camera hiccup — retry
            continue

        # ── Use the live results however the mission needs ──
        for m in result['markers']:
            print(f"[ARUCO] ID:{m['id']} "
                  f"{'VALID' if m['valid'] else 'INVALID'} "
                  f"dist={m['distance_m']}")
        for lp in result['landpads']:
            print(f"[LANDPAD] {lp['class_name']} dist={lp['distance_m']:.2f}")

        await asyncio.sleep(0.02)


async def main():
    # ── Camera: flight side owns it ────────────────────────────
    cam = RealSenseManager()
    for _ in range(10):                       # warm up (settle auto-exposure)
        try:
            cam.grab()
        except RuntimeError:
            pass

    # ── Perception: loads YOLO, takes the camera intrinsics ────
    perc = PerceptionModule(cam.K, cam_height=2.0)

    # ── Flight: connect drone ──────────────────────────────────
    drone = System()
    await drone.connect(system_address="serial:///dev/ttyS6:921600")
    async for conn in drone.core.connection_state():
        if conn.is_connected:
            break

    state = {'pos': None, 'yaw': 0.0, 'stop': False}
    mon = asyncio.create_task(position_monitor(drone, state))
    perc_task = asyncio.create_task(perception_loop(cam, perc, state))

    # ── ... the teammate's takeoff / waypoint / sweep code here ...
    await asyncio.sleep(30)   # placeholder for the actual mission

    # ── Shutdown ───────────────────────────────────────────────
    state['stop'] = True
    perc_task.cancel()
    mon.cancel()
    for t in (perc_task, mon):
        try:
            await t
        except asyncio.CancelledError:
            pass

    perc.save_outputs()       # map PNG + report
    cam.stop()                # flight side owns the camera → flight side stops it


if __name__ == "__main__":
    asyncio.run(main())
