#!/usr/bin/env python3

"""Method 4: convex-hull decongestion for multi-UAV avoidance.

Pure implementation (no ROS / Gazebo dependencies, mirroring swap_3d.py).

The "dense environment" of the paper is, in this simulation, the swarm itself:
the neighbours' keep-out cylinders. Method 4 models the surrounding flock with
the convex hull of the neighbours' horizontal positions:

  * while the drone is OUTSIDE the hull, neighbour hull edges act like solid
    keep-out boundaries: the reference velocity is the toward-goal drive plus
    a push away from the nearest hull vertex;
  * when the drone is INSIDE the hull (neighbours have boxed it in / a
    head-on crowd is converging), it is decongested through the widest angular
    gap between hull vertices as seen from the drone - a classic escape-the-
    -encirclement tactic that is cheap and hull-based.

The result is a complete 3D reference velocity (like 3D-SWAP), not an offset.
"""

import dataclasses
import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

Point = Tuple[float, float]


@dataclass
class HullParams:
    """Tunable behaviour of the convex-hull avoidance method."""

    activation: float = 2.0     # outside-hull push range [m]
    gain: float = 1.3           # outside-hull push gain [1/s]
    base_speed: float = 0.4     # toward-goal cruise speed [m/s]
    speed_cap: float = 1.0      # reference magnitude cap [m/s]
    stop_radius: float = 0.25   # considered "goal reached" below this
    keepout: float = 0.7        # safety radius used for gap-edge widening [m]


def convex_hull_xy(points: Sequence[Point]) -> List[Point]:
    """Monotone-chain 2D convex hull (CCW, no collinear interior points).

    Returns the hull vertices in counter-clockwise order. Degenerate inputs
    (0-2 points) return the input points untouched.
    """
    pts = sorted(set(points))
    if len(pts) <= 2:
        return list(pts)

    def cross(o, a, b):
        return ((a[0] - o[0]) * (b[1] - o[1])
                - (a[1] - o[1]) * (b[0] - o[0]))

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def point_in_hull(pt: Point, hull: Sequence[Point]) -> bool:
    """Ray-casting point-in-polygon test for a convex hull (CCW or any order)."""
    if len(hull) < 3:
        return False
    inside = False
    n = len(hull)
    j = n - 1
    for i in range(n):
        xi, yi = hull[i]
        xj, yj = hull[j]
        if ((yi > pt[1]) != (yj > pt[1])
                and pt[0] < (xj - xi) * (pt[1] - yi) / ((yj - yi) or 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def _hull_centroid(hull: Sequence[Point]) -> Point:
    if not hull:
        return (0.0, 0.0)
    sx = sum(p[0] for p in hull) / len(hull)
    sy = sum(p[1] for p in hull) / len(hull)
    return (sx, sy)


def _widest_gap_escape(own: Point, hull: Sequence[Point],
                       keepout: float, preferred=None) -> Point:
    """Escape unit direction through the widest angular gap of the hull.

    Angles of the hull vertices as seen from ``own``, each fattened by the
    keep-out half-width. Among the near-widest gaps the one most aligned with
    ``preferred`` (a unit direction, e.g. toward the goal) wins, so an
    encircled drone escapes through the largest AND most useful opening
    instead of an arbitrary one. Falls back to pointing away from the hull
    centroid.
    """
    cx, cy = own
    angles = sorted(math.atan2(hy - cy, hx - cx) for hx, hy in hull)
    n = len(angles)
    widest = -1.0
    best = None
    for i in range(n):
        a0, a1 = angles[i], angles[(i + 1) % n]
        if i == n - 1:
            gap = a1 + 2.0 * math.pi - a0
            a1 += 2.0 * math.pi
        else:
            gap = a1 - a0
        bisect = a0 + gap / 2.0
        ux, uy = math.cos(bisect), math.sin(bisect)
        score = gap
        if preferred is not None:
            align = max(0.0, ux * preferred[0] + uy * preferred[1])
            score = gap + 2.0 * align
        if score > widest:
            widest = score
            best = (ux, uy)
    if best is not None:
        return best
    mcx, mcy = _hull_centroid(hull)
    dx, dy = cx - mcx, cy - mcy
    d = math.hypot(dx, dy) or 1.0
    return (dx / d, dy / d)


def _clamp_magnitude(vx, vy, vz, cap, out):
    mag = math.hypot(vx, vy, vz)
    if mag > cap and mag > 1e-9:
        k = cap / mag
        return vx * k, vy * k, vz * k
    return vx, vy, vz


def method_4_convex_hull(own, others, own_goal, params=None):
    """Convex-hull decongestion reference velocity for the ego drone.

    ``own`` / ``others`` / ``own_goal`` are 3-tuples (x, y, z). Returns
    ``(vx, vy, vz)`` world-metre reference velocity.
    """
    params = params or HullParams()
    gx, gy, gz = own_goal
    ox, oy, oz = own
    dx = gx - ox
    dy = gy - oy
    dz = gz - oz
    gd = math.hypot(dx, dy, dz)

    goal_vec = (0.0, 0.0, 0.0)
    if gd > params.stop_radius:
        sp = min(params.base_speed, gd / 0.5)
        goal_vec = (dx / gd * sp, dy / gd * sp, dz / gd * sp)

    if not others:
        return goal_vec

    others_xy = [(o[0], o[1]) for o in others]
    hull = convex_hull_xy(others_xy)
    inside = point_in_hull((ox, oy), hull)

    if inside:
        pref = None
        if gd > 1e-6:
            pref = (dx / gd, dy / gd)
        ex, ey = _widest_gap_escape((ox, oy), hull, params.keepout, pref)
        vx = goal_vec[0] + ex * params.speed_cap
        vy = goal_vec[1] + ey * params.speed_cap
        vz = goal_vec[2]
        return _clamp_magnitude(vx, vy, vz, params.speed_cap, None)

    nearest, nd = None, 1e9
    for o in others:
        d = math.hypot(ox - o[0], oy - o[1])
        if d < nd:
            nearest, nd = o, d
    if nearest is None or nd > params.activation:
        return goal_vec

    ux = (ox - nearest[0]) / (nd or 1e-9)
    uy = (oy - nearest[1]) / (nd or 1e-9)
    push = params.gain * (params.activation - nd) * max(0.0, 1.0 - nd
                                                        / params.activation)
    vx = goal_vec[0] + ux * push * params.speed_cap * 2.0
    vy = goal_vec[1] + uy * push * params.speed_cap * 2.0
    vz = goal_vec[2]
    return _clamp_magnitude(vx, vy, vz, params.speed_cap, None)