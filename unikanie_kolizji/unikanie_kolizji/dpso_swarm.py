#!/usr/bin/env python3

"""Method 3: distributed particle swarm optimization (DPSO) trajectory planner.

Pure implementation (no ROS / Gazebo dependencies, mirroring swap_3d.py) of
the DPSO loop from the swarm-formation paper:

    Input: max iteration times M
    For agent i: initialize a group of particles S_i with random velocities
                 and positions;
    for episode = 1 to M do
      for particle P_k in S_i do
        if J(P_k) < J(P_{k,pbest}) then P_{k,pbest} <- P_k
        if J(P_k) < J(P_{i,gbest}) then P_{i,gbest} <- P_k
      for particle P_k in S_i do updateVelocityAndPosition(P_k)
      if receive P_{j,gbest} from other agent j then
        if J(P_{j,gbest}) < J(P_{i,gbest}) then P_{i,gbest} <- P_{j,gbest}
    broadcast P_{i,gbest} at a certain frequency

A particle's "position" is a candidate receding-horizon velocity profile: a
flat list of 3*H floats (H velocity steps in world metres/second). The
objective J() forward-simulates the profile from the drone's current position
and scores the goal distance, a keep-out penalty against the (constant-
-velocity) predicted neighbour lanes, plus control effort / smoothness.

The algorithm is receding-horizon: the swarm replans every ``replan_ticks``
node ticks (default 10 -> 0.5 s) and the drone follows its stored gbest
profile between replans. The inter-agent gbest exchange ("receive
P_{j,gbest}", "broadcast P_{i,gbest}") happens through the optional
``external_gbests`` argument: each received trajectory is blended with the
constant-velocity lane prediction inside the ego objective's keep-out term.

Neighbours that have nearly stopped (parked at their own target, e.g. a
drone that already docked) keep a much smaller clearance radius: they are
static furniture the ego drone is allowed to slip past, while *moving*
neighbours keep the full keep-out band. This is what lets the second drone
complete its own docking approach once the first has arrived.
"""

import dataclasses
import math
import random
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

Vec3 = Tuple[float, float, float]


@dataclass
class DPSOParams:
    """Tunables of the DPSO trajectory planner."""

    pop: int = 40          # particles per drone swarm S_i
    max_iter: int = 25     # max iteration times M on each replan
    horizon: int = 12      # planning horizon length H (velocity steps)
    plan_dt: float = 0.3   # seconds per profile step
    replan_ticks: int = 6  # re-run DPSO every N node ticks (~0.3 s at 20 Hz)
    tick_dt: float = 0.05   # node control period [s]
    w: float = 0.6         # PSO inertia weight
    c1: float = 1.6        # cognitive (pbest) coefficient
    c2: float = 1.6        # social (gbest) coefficient
    w_goal: float = 8.0    # J weight: running distance to goal
    term_goal: float = 2.0   # J weight: final-horizon distance to goal (soft stop)
    capture_radius: float = 2.0  # deterministic docking capture radius [m]
    w_ob: float = 9.0      # J weight: keep-out penalty to neighbours
    w_effort: float = 0.1  # J weight: |v|^2 energy cost
    w_smooth: float = 0.3  # J weight: |dv|^2 acceleration cost
    keepout: float = 1.5   # neighbour keep-out radius for J [m]
    ob_stop: float = 1.7   # beyond this distance J ignores a neighbour [m]
    hard_stop: float = 1.95  # tight-phase clearance starts inside this radius
    still_speed: float = 0.15  # below this a neighbour counts as parked [m/s]
    keepout_still: float = 0.8   # keep-out radius vs a parked neighbour [m]
    ob_stop_still: float = 1.0   # J ignore distance vs a parked neighbour [m]
    hard_stop_still: float = 1.25  # tight-phase radius vs a parked neighbour [m]
    exchange_blend: float = 0.3  # blend of exchanged plan vs linear prediction
    speed_cap: float = 0.9  # velocity magnitude cap [m/s]
    seed: Optional[int] = None
    extra_cost: Optional[Callable] = None  # combined method: J_extra(pos, t)


@dataclass
class Particle:
    """One DPSO particle: candidate profile + PSO velocity + personal best."""

    x: List[float]
    v: List[float]
    pbest: List[float]
    pbest_cost: float


@dataclass
class DPSOState:
    """Persistent per-drone planner state (receding-horizon bookkeeping)."""

    particles: List[Particle]
    gbest: List[float]
    gbest_cost: float
    replan_counter: int = 0
    plan_timer: float = 0.0
    plan_step: int = 0
    latest: Optional[Vec3] = None
    side: Optional[float] = None  # committed lateral reference (hysteresis)
    consensus_gbest: Optional[List[float]] = None  # STGO adopted plan
    consensus_owner: Optional[str] = None  # drone_id owning the adopted plan


# --- objective J ------------------------------------------------------------

def _profile_cost(x, own, others, others_vel, goal, params,
                  other_paths=None, external_weights=None) -> float:
    """J(P): forward-simulate profile ``x`` and score it.

    ``own`` / ``goal`` are 3-tuples; ``others[i]`` is neighbour position at
    the start of the horizon and ``others_vel[i]`` its assumed constant
    velocity (world m/s). ``other_paths[i]`` optionally maps neighbour ``i``
    to its predicted position per horizon step (the exchanged neighbour
    plan). ``external_weights[i]`` overrides ``params.exchange_blend`` for
    that neighbour (used by STGO to give the adopted plan full weight).
    Returns a low-is-better scalar cost.
    """
    h, dt = params.horizon, params.plan_dt
    if others_vel is None:
        others_vel = [(0.0, 0.0, 0.0)] * len(others)
    cost = 0.0
    px, py, pz = own
    pvx = pvy = pvz = 0.0
    for t in range(h):
        vx = x[3 * t]
        vy = x[3 * t + 1]
        vz = x[3 * t + 2]
        px += vx * dt
        py += vy * dt
        pz += vz * dt
        cost += params.w_goal * (math.hypot(px - goal[0], py - goal[1],
                                            pz - goal[2]))
        for i in range(len(others)):
            ospeed = math.hypot(*others_vel[i]) if others_vel else 0.0
            ob = (params.ob_stop_still if ospeed < params.still_speed
                  else params.ob_stop)
            if other_paths and other_paths[i]:
                ex, ey, ez = other_paths[i][t]
                ox = others[i][0] + others_vel[i][0] * dt * t
                oy = others[i][1] + others_vel[i][1] * dt * t
                oz = others[i][2] + others_vel[i][2] * dt * t
                b = params.exchange_blend
                if external_weights is not None \
                        and i < len(external_weights) \
                        and external_weights[i] is not None:
                    b = external_weights[i]
                b = min(max(b, 0.0), 1.0)
                ox, oy, oz = (b * ex + (1.0 - b) * ox,
                              b * ey + (1.0 - b) * oy,
                              b * ez + (1.0 - b) * oz)
            else:
                ox = others[i][0] + others_vel[i][0] * dt * t
                oy = others[i][1] + others_vel[i][1] * dt * t
                oz = others[i][2] + others_vel[i][2] * dt * t
            d = math.hypot(px - ox, py - oy, pz - oz)
            if d < ob:
                cost += (params.w_ob
                         * ((ob - d) ** 2) / ob)
        cost += params.w_effort * (vx * vx + vy * vy + vz * vz)
        cost += params.w_smooth * ((vx - pvx) ** 2 + (vy - pvy) ** 2
                                   + (vz - pvz) ** 2)
        pvx, pvy, pvz = vx, vy, vz
        if params.extra_cost is not None:
            cost += params.extra_cost((px, py, pz), t)
    if params.term_goal > 0.0:
        cost += (params.term_goal * (math.hypot(px - goal[0], py - goal[1],
                                                pz - goal[2])))
    return cost


# --- PSO mechanics ----------------------------------------------------------

def _goal_profile(own, goal, params) -> List[float]:
    """Seed profile: straight toward the goal at base speed, repeated."""
    gx, gy, gz = goal
    dx = gx - own[0]
    dy = gy - own[1]
    dz = gz - own[2]
    d = math.hypot(dx, dy, dz) or 1.0
    sp = min(0.5, params.speed_cap)
    step = [dx / d * sp, dy / d * sp, dz / d * sp]
    return list(step * params.horizon)


def _with_goal_seed(particles, own, goal, params) -> List[Particle]:
    """Inject the straight-to-goal particle on every replan.

    The warm start (last gbest) can get wedged in a stale basin right after a
    crossing detour, at which point Gaussian perturbations around it cannot
    bridge the velocity reversal needed to aim at the goal again. Re-seeding
    the goal profile guarantees the nominal optimum is always present in the
    initial population, so the swarm can never permanently strand a drone.
    """
    return _with_extra_seed(particles, _goal_profile(own, goal, params),
                            params)


def _with_extra_seed(particles, profile, params) -> List[Particle]:
    """Inject ``profile`` (a velocity profile) as a fresh particle.

    STGO uses this to seed every non-owner drone's population with the
    adopted consensus plan: the search then re-optimises the globally
    cheapest plan toward its own goal instead of being trapped in a stale
    near-zero basin.
    """
    dim = 3 * params.horizon
    profile = list(profile[:dim])
    seed = Particle(x=profile, v=[0.0] * dim,
                    pbest=profile, pbest_cost=1e18)
    particles.append(seed)
    return particles


def _build_particles(warm, params, rnd) -> List[Particle]:
    """Spawn ``pop`` particles by perturbing the warm-start profile."""
    dim = 3 * params.horizon
    noise = 0.45 * params.speed_cap
    particles = []
    for _ in range(params.pop):
        x = [min(max(v + rnd.gauss(0.0, noise), -params.speed_cap),
                 params.speed_cap) for v in warm]
        particles.append(Particle(x=x, v=[0.0] * dim, pbest=list(x),
                                  pbest_cost=1e18))
    return particles


def _prep(warm, own, others, others_vel, goal, params, rnd) -> List[Particle]:
    if warm is None:
        warm = _goal_profile(own, goal, params)
    return _build_particles(warm, params, rnd)


def _clamp_steps(x: List[float], params) -> None:
    """Clamp every profile step's velocity magnitude to ``speed_cap``."""
    for t in range(params.horizon):
        vx = x[3 * t]
        vy = x[3 * t + 1]
        vz = x[3 * t + 2]
        mag = math.hypot(vx, vy, vz)
        if mag > params.speed_cap and mag > 1e-9:
            k = params.speed_cap / mag
            x[3 * t] *= k
            x[3 * t + 1] *= k
            x[3 * t + 2] *= k


def _one_pso_pass(state, own, others, others_vel, goal, params, rnd,
                  other_paths=None, external_weights=None) -> None:
    """Run the paper's inner loop for ``max_iter`` episodes."""
    dim = 3 * params.horizon
    swarm_best = state.gbest
    for _ in range(params.max_iter):
        for p in state.particles:
            c = _profile_cost(p.x, own, others, others_vel, goal, params,
                              other_paths, external_weights)
            if c < p.pbest_cost:
                p.pbest_cost = c
                p.pbest = list(p.x)
            if c < state.gbest_cost:
                state.gbest_cost = c
                state.gbest = list(p.x)
                swarm_best = state.gbest
        for p in state.particles:
            for i in range(dim):
                r1 = rnd.random()
                r2 = rnd.random()
                p.v[i] = (params.w * p.v[i]
                          + params.c1 * r1 * (p.pbest[i] - p.x[i])
                          + params.c2 * r2 * (swarm_best[i] - p.x[i]))
            for i in range(dim):
                p.x[i] += p.v[i]
            _clamp_steps(p.x, params)


# --- public entry point -----------------------------------------------------

def method_3_dpso(own, others, own_goal, params=None, state=None,
                  others_vel=None, external_gbests=None,
                  external_weights=None, extra_seed=None):
    """DPSO reference velocity for the ego drone.

    ``own`` / ``others`` / ``own_goal`` are 3-tuples in world metres.
    Returns ``(vx, vy, vz, new_state)``: pass ``new_state`` back on the next
    call to keep the receding horizon running (``None`` starts fresh).
    """
    params = params or DPSOParams()
    rnd = random.Random(params.seed)
    if others_vel is None:
        others_vel = [(0.0, 0.0, 0.0)] * len(others)

    # Paper's inter-agent exchange: convert a neighbour's gbest plan into its
    # predicted trajectory, which J then treats like a moving obstacle. This
    # sharpens the keep-out model without ever overwriting our own goal.
    other_paths = None
    if external_gbests:
        other_paths = [None] * len(others)
        for j, gb in enumerate(external_gbests[:len(others)]):
            if gb is None or len(gb) < 3 * params.horizon:
                continue
            pts = []
            ox, oy, oz = others[j]
            for t in range(params.horizon):
                pts.append((ox, oy, oz))
                ox += gb[3 * t] * params.plan_dt
                oy += gb[3 * t + 1] * params.plan_dt
                oz += gb[3 * t + 2] * params.plan_dt
            other_paths[j] = pts

    if state is None:
        state = DPSOState(
            particles=[], gbest=_goal_profile(own, own_goal, params),
            gbest_cost=1e18)
        # Initial swarm + first full pass (the paper's initialization step).
        state.particles = _prep(state.gbest, own, others, others_vel,
                                own_goal, params, rnd)
        _with_goal_seed(state.particles, own, own_goal, params)
        if extra_seed is not None:
            _with_extra_seed(state.particles, extra_seed, params)
        _one_pso_pass(state, own, others, others_vel, own_goal, params, rnd,
                      other_paths, external_weights)

    # Replan on cadence (or when the profile has been consumed).
    if (state.replan_counter >= params.replan_ticks
            or state.plan_step >= params.horizon - 1
            or not state.particles):
        state.particles = _prep(state.gbest, own, others, others_vel,
                                own_goal, params, rnd)
        _with_goal_seed(state.particles, own, own_goal, params)
        if extra_seed is not None:
            _with_extra_seed(state.particles, extra_seed, params)
        # gbest_cost is cached across the drone's lifetime while the world
        # position changes every tick; a stale low value would let any plan
        # freeze permanently. Re-evaluate the warm start at the new state.
        state.gbest_cost = _profile_cost(state.gbest, own, others,
                                         others_vel, own_goal, params,
                                         other_paths, external_weights)
        _one_pso_pass(state, own, others, others_vel, own_goal, params, rnd,
                      other_paths, external_weights)
        state.replan_counter = 0
        state.plan_step = 0
        state.plan_timer = 0.0
    else:
        state.replan_counter += 1

    return extract_and_postprocess(
        state, own, others, own_goal, params, others_vel=others_vel)


def extract_and_postprocess(state, own, others, own_goal, params,
                            others_vel=None):
    """Extract velocity from the gbest and apply safety overrides.

    Shared by the normal DPSO path and the STGO adoption path.  Applies
    capture-radius landing, tight-phase clearance, directional hysteresis
    and speed capping.  Returns ``(vx, vy, vz, state)``.
    """
    if others_vel is None:
        others_vel = [(0.0, 0.0, 0.0)] * len(others)

    vx, vy, vz = state.gbest[0], state.gbest[1], state.gbest[2]
    if state.plan_step < params.horizon:
        i = 3 * state.plan_step
        vx, vy, vz = state.gbest[i], state.gbest[i + 1], state.gbest[i + 2]
    # Capture-radius landing: inside the capture sphere the reference is a
    # deterministic proportional glide to the goal, so the receding horizon
    # always docks (no overshoot) regardless of plan quality.
    gd = math.hypot(own[0] - own_goal[0], own[1] - own_goal[1],
                    own[2] - own_goal[2])
    if gd < params.capture_radius and gd > 1e-9:
        k = params.speed_cap * (gd / params.capture_radius) / gd
        vx = (own_goal[0] - own[0]) * k
        vy = (own_goal[1] - own[1]) * k
        vz = (own_goal[2] - own[2]) * k
    # Tight-phase clearance guarantee: treat J's keep-out as a hard, always
    # satisfiable constraint on the emitted first step. Inside the keep-out
    # radius the reference becomes a direct radial escape (blended smoothly
    # through the approach band down to hard_stop), so a noisy or slightly
    # stale plan can never let the gap collapse. Parked neighbours use the
    # smaller "still" envelope so they act as passable furniture.
    for j, (ox, oy, oz) in enumerate(others):
        d = math.hypot(own[0] - ox, own[1] - oy, own[2] - oz)
        still = (math.hypot(*others_vel[j]) < params.still_speed)
        hs = params.hard_stop_still if still else params.hard_stop
        ko = params.keepout_still if still else params.keepout
        if d < hs and d > 1e-9:
            band = max(hs - ko, 1e-6)
            radial = (own[0] - ox) / d, (own[1] - oy) / d, (own[2] - oz) / d
            share = min(1.0, max(0.0, (hs - d) / band))
            vx = radial[0] * params.speed_cap * share + vx * (1.0 - share)
            vy = radial[1] * params.speed_cap * share + vy * (1.0 - share)
            vz = radial[2] * params.speed_cap * share + vz * (1.0 - share)
    mag = math.hypot(vx, vy, vz)
    if mag > params.speed_cap and mag > 1e-9:
        k = params.speed_cap / mag
        vx, vy, vz = vx * k, vy * k, vz * k
    # Directional hysteresis (side commitment): in a symmetric chokepoint the
    # tight-phase escape has two near-equal passing sides; with no memory the
    # reference flips side every replan and the drone zig-zags in place. Pull a
    # flip back toward the committed lateral reference so the drone commits and
    # clears; a persistently better opposite side still wins after a few ticks.
    if state.side is not None and state.side * vx < 0.0:
        vx = 0.5 * vx + 0.5 * state.side
    state.side = vx if state.side is None else 0.85 * state.side + 0.15 * vx
    state.gbest[0:3] = [vx, vy, vz]
    state.latest = (vx, vy, vz)

    # Advance the receding-horizon pointer by the node tick.
    state.plan_timer += params.tick_dt
    while state.plan_timer >= params.plan_dt and state.plan_step < params.horizon - 1:
        state.plan_timer -= params.plan_dt
        state.plan_step += 1
    return vx, vy, vz, state