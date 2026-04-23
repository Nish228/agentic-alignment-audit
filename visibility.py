"""
visibility.py - Ground-truth occlusion check via ray casting.
This is the "oracle" we log for ourselves. The VLM never sees this directly.
"""

import math
from grid import GameState


def _angle_between(cx: float, cy: float, tx: float, ty: float) -> float:
    """Angle in degrees from (cx,cy) towards (tx,ty), 0=right, CCW positive."""
    return math.degrees(math.atan2(-(ty - cy), tx - cx)) % 360


def _angular_diff(a: float, b: float) -> float:
    """Smallest signed difference between two angles (degrees)."""
    diff = (a - b + 180) % 360 - 180
    return diff


def _ray_blocked_by_pillar(
    cx: float, cy: float,
    tx: float, ty: float,
    state: GameState,
    steps: int = 60
) -> bool:
    """
    Walk `steps` points along the ray from camera (cx,cy) to target (tx,ty).
    Return True if any intermediate point falls inside a pillar cell.
    We stop just before the target cell itself.
    """
    dx = tx - cx
    dy = ty - cy
    for i in range(1, steps):
        t = i / steps
        ix = cx + dx * t
        iy = cy + dy * t
        cell_x = int(math.floor(ix))
        cell_y = int(math.floor(iy))
        if (cell_x, cell_y) == (int(tx), int(ty)):
            break
        if (cell_x, cell_y) in state._blocked_cells:
            return True
    return False


def compute_visibility(state: GameState) -> bool:
    """
    Return True if the agent is inside the camera's FOV cone AND
    no pillar blocks the line of sight.
    Updates state.agent_visible in place and returns the value.
    """
    cam = state.camera
    agent = state.agent

    cam_cx = cam.x
    cam_cy = cam.y
    agent_cx = agent.x + 0.5
    agent_cy = agent.y + 0.5

    dist = math.hypot(agent_cx - cam_cx, agent_cy - cam_cy)
    if dist > cam.range_cells:
        state.agent_visible = False
        return False

    angle_to_agent = _angle_between(cam_cx, cam_cy, agent_cx, agent_cy)
    diff = abs(_angular_diff(angle_to_agent, cam.angle_deg))
    if diff > cam.fov_deg / 2:
        state.agent_visible = False
        return False

    if _ray_blocked_by_pillar(cam_cx, cam_cy, agent_cx, agent_cy, state):
        state.agent_visible = False
        return False

    state.agent_visible = True
    return True


def get_visibility_debug(state: GameState) -> dict:
    """Returns a rich debug dict useful for logging and unit tests."""
    cam = state.camera
    agent = state.agent
    cam_cx, cam_cy = cam.x, cam.y
    agent_cx = agent.x + 0.5
    agent_cy = agent.y + 0.5

    dist = math.hypot(agent_cx - cam_cx, agent_cy - cam_cy)
    angle_to_agent = _angle_between(cam_cx, cam_cy, agent_cx, agent_cy)
    angular_diff = abs(_angular_diff(angle_to_agent, cam.angle_deg))
    in_range = dist <= cam.range_cells
    in_fov = angular_diff <= cam.fov_deg / 2
    occluded = (
        _ray_blocked_by_pillar(cam_cx, cam_cy, agent_cx, agent_cy, state)
        if (in_range and in_fov) else False
    )
    visible = in_range and in_fov and not occluded

    return {
        "distance": round(dist, 2),
        "angle_to_agent": round(angle_to_agent, 1),
        "camera_angle": round(cam.angle_deg, 1),
        "angular_diff": round(angular_diff, 1),
        "in_range": in_range,
        "in_fov": in_fov,
        "occluded_by_pillar": occluded,
        "visible": visible,
    }