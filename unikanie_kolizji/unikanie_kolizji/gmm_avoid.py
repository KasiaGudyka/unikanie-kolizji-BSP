#!/usr/bin/env python3

"""Method 5: Gaussian-mixture-model (GMM) density avoidance for UAV swarms.

Pure implementation (no ROS / Gazebo dependencies, mirroring swap_3d.py).

The neighbours of the ego drone are modelled as the means of an isotropic
Gaussian mixture (a kernel-density estimate with Gaussian kernels). The
mixture density,

    p(x) = sum_k exp( -||x - mu_k||^2 / 2 sigma^2 ),

peaks where the swarm is densest; its negative gradient therefore points away
from the heaviest drone concentration:

    -grad p(x) = sum_k exp( ... ) * (x - mu_k) / sigma^2 .

The reference velocity combines the toward-goal drive with this repulsion,
scaled by how strongly the drone is inside the mixture support, and is capped
at the airframe speed limit. Stateless per call.
"""

import dataclasses
import math
from dataclasses import dataclass
from typing import Sequence, Tuple


@dataclass
class GMMParams:
    """Tunable behaviour of the GMM avoidance method."""

    sigma: float = 0.8         # isotropic Gaussian kernel width [m]
    gain: float = 1.4          # density-repulsion gain [1/s]
    activation: float = 2.0    # beyond this far the density push is ignored
    base_speed: float = 0.4    # toward-goal cruise speed [m/s]
    speed_cap: float = 1.0     # reference magnitude cap [m/s]
    stop_radius: float = 0.25  # considered "goal reached" below this


def _clamp_magnitude(vx, vy, vz, cap):
    mag = math.hypot(vx, vy, vz)
    if mag > cap and mag > 1e-9:
        k = cap / mag
        return vx * k, vy * k, vz * k
    return vx, vy, vz


def gmm_density_gradient(pos, others, sigma):
    """Mixture density value and its negative gradient at ``pos``.

    Returns ``(density, -grad dx, -grad dy)`` in the horizontal plane; the
    repulsion direction is ``(-grad)`` normalised on demand by the caller.
    """
    density = 0.0
    rx = ry = 0.0
    inv = 1.0 / (2.0 * sigma * sigma)
    for o in others:
        dx = pos[0] - o[0]
        dy = pos[1] - o[1]
        w = math.exp(-(dx * dx + dy * dy) * inv)
        density += w
        rx += dx * w / (sigma * sigma)
        ry += dy * w / (sigma * sigma)
    return density, rx, ry


def method_5_gmm(own, others, own_goal, params=None):
    """GMM density-avoidance reference velocity for the ego drone.

    ``own`` / ``others`` / ``own_goal`` are 3-tuples (x, y, z). Returns
    ``(vx, vy, vz)`` world-metre reference velocity.
    """
    params = params or GMMParams()
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

    density, rx, ry = gmm_density_gradient((ox, oy), others, params.sigma)
    if density < 1e-9:
        return goal_vec

    rmag = math.hypot(rx, ry)
    if rmag < 1e-9:
        return goal_vec

    push = params.gain * math.tanh(density)
    vx = goal_vec[0] + rx / rmag * push * params.speed_cap
    vy = goal_vec[1] + ry / rmag * push * params.speed_cap
    vz = goal_vec[2]
    return _clamp_magnitude(vx, vy, vz, params.speed_cap)