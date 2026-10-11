#!/usr/bin/env python3

"""3D-SWAP: decentralized 3D collision avoidance for multiple UAVs.

Pure implementation (no ROS / Gazebo dependencies) of the 3D-SWAP algorithm
from Ferrera et al., "Decentralized 3D Collision Avoidance for Multiple UAVs
in Outdoor Environments", Sensors 2018, 18, 4101.

Scope: only the inter-UAV (non-cooperative teammate) behaviour is implemented.
The LIDAR / pointcloud path of the original paper is intentionally omitted:
external obstacles are not modelled in the simulation, and the user asked to
focus on collision avoidance *between* the drones.

Each UAV keeps three concentric, world-axis-aligned cylinders:
  * collision cylinder   - radius rc, height hc : actual occupied space.
  * reserved cylinder    - radius rr, height hr : xy-conflict detection
                           (a neighbor enters conflict when its reserved
                           cylinder overlaps this one).
  * blocking cylinder    - radius rr, height hb : z-conflict detection
                           (top/bottom slabs stacked on the reserved one).

The Cylindrical Obstacle Diagram (COD) is a polar histogram of nearest-obstacle
distances. A neighbor UAV is inserted as a virtual circle of radius rc; it only
enters the COD when both reserved cylinders overlap vertically (|dz| <= hr).

State machine (two independent axes, per Algorithm 1 of the paper):
  s_xy in {xy-free, rendezvous, xy-blocked}
  s_z  in {z-free, z-blocked}

The reference velocity is a full 3D vector (vx, vy, vz) that replaces the
toward-goal velocity while a conflict is active (it is not an "offset").
"""

import dataclasses
import itertools
import math
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

TWO_PI = 2.0 * math.pi


@dataclass
class SWAP3DParams:
    """Tunable geometry / behaviour of 3D-SWAP (scaled for this simulation)."""

    rc: float = 0.35          # collision cylinder radius [m] (45cm-box drone)
    hc: float = 0.22          # collision cylinder height [m]
    rr: float = 0.9           # reserved / blocking cylinder radius [m]
    hr: float = 2.0           # reserved cylinder height [m] (vertical gate)
    hb: float = 3.0           # blocking cylinder height [m]
    n_phi: int = 360          # COD angular discretization (bins)
    va: float = 0.7           # avoidance (rendezvous) speed [m/s] (<= vmax)
    vmax: float = 1.0         # cruising speed cap [m/s]
    kref: float = 1.0         # computeRefSpeed proportional gain [1/s]
    stop_radius: float = 0.25  # considered "goal reached" below this dist
    z_stop: float = 0.1        # vertical stop tolerance [m] (altitude is tighter)
    min_sep: float = 0.8       # safety net: hard repulsion below 2*rc (approx)
    waypoint_blend: float = 0.25  # forward (toward-goal) bias during the
                                  # roundabout: WP dominates sideways, this term
                                  # keeps the drone making progress while it rounds

    def xy_conflict_threshold(self) -> float:
        """Neighbor is in xy-conflict when COD distance <= 2*rr - rc."""
        return 2.0 * self.rr - self.rc

    def sanitized(self) -> 'SWAP3DParams':
        """Return a copy with invariants enforced (va never exceeds vmax)."""
        return dataclasses.replace(self, va=min(self.va, self.vmax))


@dataclass
class SWAPResult:
    """One reference-velocity verdict for a single UAV."""

    s_xy: str                      # 'xy-free' | 'rendezvous' | 'xy-blocked'
    s_z: str                       # 'z-free' | 'z-blocked'
    vx: float                      # commanded world-frame velocity    [m/s]
    vy: float
    vz: float
    phi_a: Optional[float] = None  # selected avoidance bearing [rad]
    phi_c: Optional[float] = None   # conflict bearing used to select phi_a
    conflicts: int = 0             # number of active xy-conflicts
    d_xy: float = 0.0              # planar distance to nearest neighbor


def _wrap(angle: float) -> float:
    """Normalize an angle to [0, 2*pi)."""
    ang = math.fmod(angle, TWO_PI)
    if ang < 0.0:
        ang += TWO_PI
    return ang


def ang_diff(a: float, b: float) -> float:
    """Smallest absolute angular difference between a and b in [0, pi]."""
    d = abs(_wrap(a) - _wrap(b))
    return min(d, TWO_PI - d)


class CylindricalObstacleDiagram:
    """Polar histogram of nearest-obstacle distances (COD of the paper).

    Only 'dynamic' (moving UAV) obstacles are stored. A neighbor is inserted as
    a virtual circle of 'radius' centred at its bearing; the bins covered by
    that circle's angular footprint get its closest-plane distance.
    """

    def __init__(self, params: SWAP3DParams, label: Optional[str] = None):
        self.params = params
        self.label = label
        self._rho: List[float] = []
        self._dynamic: List[bool] = []
        self._obstacles: List[Tuple[float, float]] = []
        self.reset()

    def reset(self) -> None:
        self._rho = [math.inf] * self.params.n_phi
        self._dynamic = [False] * self.params.n_phi
        self._obstacles = []

    def add_round_obstacle(self, rho_xy: float, phi: float, radius: float,
                           z_gate_ok: bool) -> None:
        """Insert a circular obstacle of 'radius' at planar distance rho_xy.

        'z_gate_ok' must already encode the vertical gate (|dz| <= hr): when
        False the obstacle always stays in the reserved cylinder, i.e. it is
        not a xy-conflict candidate.
        """
        if not z_gate_ok or rho_xy < 1e-9:
            return
        closest = max(rho_xy - radius, 1e-4)
        self._obstacles.append((closest, _wrap(phi)))
        half_span = math.asin(min(radius / rho_xy, 1.0))
        cell = 2.0 * math.pi / self.params.n_phi
        n_cover = int(math.ceil(half_span / cell))
        center = int(round(_wrap(phi) / cell)) % self.params.n_phi
        for k in range(center - n_cover, center + n_cover + 1):
            idx = k % self.params.n_phi
            if closest < self._rho[idx]:
                self._rho[idx] = closest
                self._dynamic[idx] = True

    def conflict_angles(self) -> Tuple[List[Tuple[float, float]], Optional[int]]:
        """Merge conflicted bins into sectors; return (sectors, n_conflicts).

        Each sector is (rho_min, phi) where phi is the bearing of the closest
        conflicted obstacle centre (the conflict angle phi_c of the paper).
        A sector may wrap around the 0/2pi seam.
        """
        threshold = self.params.xy_conflict_threshold()
        flags = [self._dynamic[i] and self._rho[i] <= threshold
                 for i in range(self.params.n_phi)]
        n_conflicts = sum(flags)
        if not n_conflicts:
            return [], 0

        cell = TWO_PI / self.params.n_phi
        sectors = []
        for run in _circular_runs(flags):
            runset = set(run)
            best: Optional[Tuple[float, float]] = None
            best_rho = math.inf
            for closest, phi_c in self._obstacles:
                if closest > threshold:
                    continue
                phi_ok = self._sector_hits(phi_c, closest, runset)
                if phi_ok and closest < best_rho:
                    best = (closest, phi_c)
                    best_rho = closest
            if best is None:  # fallback: closest conflicted bin in the run
                rhos = [self._rho[i] for i in run]
                k = rhos.index(min(rhos))
                best = (rhos[k], _wrap(run[k] * cell))
            sectors.append(best)
        return sectors, n_conflicts

    def _sector_hits(self, phi_c: float, closest: float,
                     runset: set) -> bool:
        """True when this obstacle's swept angular band touches the run."""
        cell = TWO_PI / self.params.n_phi
        n = self.params.n_phi
        center = int(round(phi_c / cell)) % n
        half = int(math.ceil(
            math.asin(min(self.params.rc / (closest + self.params.rc), 1.0))
            / cell))
        return any((k % n) in runset
                   for k in range(center - half, center + half + 1))


def _circular_runs(flags: Sequence[bool]) -> List[List[int]]:
    """Split a circular boolean array into maximal runs of True (wrapping)."""
    n = len(flags)
    if not any(flags):
        return []
    start = next((i for i in range(n) if flags[i]), 0)
    while flags[(start - 1) % n]:
        start = (start - 1) % n
    runs = []
    run: List[int] = []
    for offset in range(n):
        i = (start + offset) % n
        if flags[i]:
            run.append(i)
        elif run:
            runs.append(run)
            run = []
    if run:
        runs.append(run)
    return runs


def compute_ref_speed(distance: float, vmax: float, kref: float,
                      stop_radius: float) -> float:
    """Proportional-then-saturated reference speed (paper Line 16/23)."""
    if distance <= stop_radius:
        return 0.0
    return min(vmax, kref * distance)


def _pick_avoidance_direction(conflicts: Sequence[Tuple[float, float]],
                              params: SWAP3DParams) -> Optional[float]:
    """Select phi_a = phi_c - pi/2 (counter-clockwise rule) free of sectors.

    Conflicts are visited closest-first. Returns None when every candidate is
    inside another conflict's forbidden sector (xy-blocked).
    """
    ordered = sorted(conflicts, key=lambda c: c[0])
    forbidden = [(math.fmod(phi - math.pi / 2, TWO_PI),
                  math.fmod(phi + math.pi / 2, TWO_PI)) for _, phi in ordered]
    for rho, phi_c in ordered:
        candidate = math.fmod(phi_c - math.pi / 2, TWO_PI)
        if all(ang_diff(candidate, ophi_c) >= math.pi / 2 - 1e-6
               for _, ophi_c in ordered):
            return candidate if candidate >= 0.0 else candidate + TWO_PI
    return None


def _z_blocked_by(own_z: float, other_z: float, d_xy: float,
                  params: SWAP3DParams) -> bool:
    """True when the other UAV blocks vertical motion through a z-conflict.

    Paper Definition 3: B+ of one UAV intersecting B- of the other requires
    hr <= |dz| <= hb AND the blocking cylinders overlapping horizontally
    (d_xy < 2*rr).
    """
    dz = abs(other_z - own_z)
    return d_xy < 2.0 * params.rr and params.hr <= dz <= params.hb


def compute_reference_velocity(
        own_pos: Sequence[float], goal_pos: Sequence[float],
        others: Sequence[Sequence[float]],
        params: Optional[SWAP3DParams] = None) -> SWAPResult:
    """Run one 3D-SWAP iteration for the ego UAV.

    own_pos / goal_pos / others entries are (x, y, z) world coordinates in
    metres. Only member-to-member avoidance is considered (no pointcloud).
    """
    params = (params or SWAP3DParams()).sanitized()
    ox, oy, oz = own_pos
    gx, gy, gz = goal_pos
    dg_x = gx - ox
    dg_y = gy - oy
    dg_z = gz - oz
    dg_xy = math.hypot(dg_x, dg_y)
    phi_goal = _wrap(math.atan2(dg_y, dg_x))

    cod = CylindricalObstacleDiagram(params)
    nearest = float('inf')
    for other in others:
        dx, dy, dz = other[0] - ox, other[1] - oy, other[2] - oz
        rho = math.hypot(dx, dy)
        nearest = min(nearest, rho)
        cod.add_round_obstacle(rho, _wrap(math.atan2(dy, dx)), params.rc,
                               z_gate_ok=abs(dz) <= params.hr)

    conflicts, n_conflicts = cod.conflict_angles()

    # ---- Safety net: collision about to happen anyway (paper Sec. 5.2).
    # Push directly away from ALL neighbours (averaged) at full vmax so the
    # escape beats the closure rate; a single-neighbour push can be defeated
    # by a third drone converging from the other side.
    if nearest < 2.0 * params.rc - 1e-3:
        sx = sy = 0.0
        for other in others:
            dx = ox - other[0]
            dy = oy - other[1]
            d = math.hypot(dx, dy)
            if d > 1e-6:
                sx += dx / d
                sy += dy / d
        n = math.hypot(sx, sy)
        if n < 1e-6:
            sx, sy, n = 1.0, 0.0, 1.0
        sx, sy = sx / n, sy / n
        return SWAPResult('rendezvous', 'z-free',
                          sx * params.vmax, sy * params.vmax, 0.0,
                          phi_a=_wrap(math.atan2(sy, sx)), phi_c=None,
                          conflicts=n_conflicts, d_xy=nearest)

    # ---- Horizontal state machine.
    s_xy = 'xy-free'
    vx = vy = 0.0
    phi_a = phi_c = None
    if conflicts:
        blocked = any(ang_diff(phi_goal, phi) < math.pi / 2 - 1e-6
                      for _, phi in conflicts)
        if blocked:
            phi_a = _pick_avoidance_direction(conflicts, params)
            if phi_a is not None:
                s_xy = 'rendezvous'
                phi_c = min(conflicts, key=lambda c: c[0])[1]
                # Roundabout with a bounded forward bias: the escape bearing
                # (phi_a) stays dominant, a fraction of the goal bearing is
                # blended in so the drone still advances toward its waypoint
                # while rounding the conflict instead of orbiting it in place.
                dx = (math.cos(phi_a)
                      + params.waypoint_blend * math.cos(phi_goal))
                dy = (math.sin(phi_a)
                      + params.waypoint_blend * math.sin(phi_goal))
                n = math.hypot(dx, dy)
                if n < 1e-6:
                    dx, dy, n = math.cos(phi_a), math.sin(phi_a), 1.0
                vx = dx / n * params.va
                vy = dy / n * params.va
            else:
                s_xy = 'xy-blocked'
        else:
            s_xy = 'xy-free'
    if s_xy == 'xy-free':
        speed = compute_ref_speed(dg_xy, params.vmax, params.kref,
                                  params.stop_radius)
        vx = math.cos(phi_goal) * speed
        vy = math.sin(phi_goal) * speed

    # ---- Vertical state machine (z-conflicts that block our path).
    s_z = 'z-free'
    vz = 0.0
    desired_dir = 1.0 if dg_z >= 0.0 else -1.0
    blocking = False
    for other in others:
        if _z_blocked_by(oz, other[2],
                         math.hypot(other[0] - ox, other[1] - oy), params):
            other_side = 1.0 if other[2] > oz else -1.0
            if desired_dir == other_side:
                blocking = True
                break
    if blocking:
        s_z = 'z-blocked'
    else:
        vz = desired_dir * compute_ref_speed(abs(dg_z), params.vmax,
                                             params.kref, params.z_stop)

    return SWAPResult(s_xy, s_z, vx, vy, vz, phi_a=phi_a, phi_c=phi_c,
                      conflicts=n_conflicts, d_xy=nearest)