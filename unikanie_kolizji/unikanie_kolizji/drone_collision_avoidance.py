#!/usr/bin/env python3

import json
import math
import os
import statistics
import time
import traceback

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix

from unikanie_kolizji import combined_avoid
from unikanie_kolizji import convex_hull_avoid
from unikanie_kolizji import dpso_swarm
from unikanie_kolizji import gmm_avoid
from unikanie_kolizji import swap_3d


def _pct(values, p):
    """Nearest-rank percentile of ``values`` (``None`` when empty)."""
    if not values:
        return None
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, int(round(p / 100 * (len(ordered) - 1)))))
    return ordered[idx]


def scale_authority(cmd, authority):
    """Scale an avoidance command's linear magnitude by ``authority`` (0..1).

    Airframe-aware authority: heavy drones are commanded at fractionally
    gentler avoidance speeds (their real airframes are far less agile than a
    toy) while the command direction and any zero-at-goal behaviour are
    preserved (0 * authority == 0). Authority 1.0 is the identity.
    """
    if authority == 1.0:
        return cmd
    scaled = Twist()
    scaled.linear.x = cmd.linear.x * authority
    scaled.linear.y = cmd.linear.y * authority
    scaled.linear.z = cmd.linear.z * authority
    scaled.angular.x = cmd.angular.x
    scaled.angular.y = cmd.angular.y
    scaled.angular.z = cmd.angular.z
    return scaled


# The six collision avoidance methods:
#   method_1 (GPS/no-fly-zone repulsion) returns an avoidance Twist offset per
#             drone;
#   method_2 (3D-SWAP) returns a FULL reference velocity that replaces the base
#             controller's toward-goal velocity;
#   method_3 (DPSO), method_4 (convex hull), method_5 (GMM) and method_6
#             (combined DPSO+GMM+hull) also return FULL reference velocities,
#             computed by the pure planners in dpso_swarm.py / convex_hull_avoid.py /
#             gmm_avoid.py / combined_avoid.py.
#
# All methods are N-generic: method_1 sums the repulsion from all other drones,
# method_2 feeds every other drone into the 3D-SWAP COD as an obstacle, and the
# full-reference planners accept any number of neighbours.

def method_1_no_flight_zone(own_pos, other_pos, no_flight_radius=0.75,
                            safe_margin=0.15, gain=1.5):
    """GPS/no-fly-zone repulsion (implemented).

    Continuous falloff: the repulsion rises smoothly with approach instead of
    switching on/off at a single radius (the old hard threshold excited the
    rotor/PID lag and made crossing drones re-converge — the "glued" symptom).
    Pushes the drone away from the other drone when inside the release ring.
    Returns a Twist offset suitable for the drone's cmd_vel topic.
    """
    if own_pos is None or other_pos is None:
        return Twist()
    cmd, _ = method_1_field(own_pos, [other_pos],
                            no_flight_radius=no_flight_radius,
                            safe_margin=safe_margin, gain=gain)
    return cmd


def _repulsion_magnitude(dist, release_dist, gain):
    """Continuous, zero at ``release_dist``, rising as the drones close in."""
    if dist >= release_dist or dist < 1e-6:
        return 0.0
    return gain * (release_dist - dist)


def method_1_field(own_pos, others, anchor=None, no_flight_radius=0.75,
                   safe_margin=0.15, gain=1.5, activate=1.15, release=1.30):
    """GPS/no-fly-zone repulsion with a continuous field + direction latch.

    Two flaws of the plain per-pair repulsion caused the "glued"/deadlock
    behaviour on mirrored crossing paths: (1) a hard activation step and
    (2) no memory, so each time the goal-seeking command pulled the pair back
    together the escape direction could flip. This version keys the field
    smoothly to distance and latches the escape direction once committed:

      * a fresh engagement only starts inside ``activate`` x the no-fly
        threshold (so drones at spawn, already 2 m apart, are left alone);
      * the committed direction is held until the pair separates past
        ``release`` x the threshold or the field genuinely changes sign
        (e.g. after passing on the other side).

    Returns ``(Twist, new_anchor)`` where ``new_anchor`` is the latched unit
    escape direction (or ``None`` when idle) to feed back on the next tick.
    """
    cmd = Twist()
    if own_pos is None or not others:
        return cmd, None

    threshold = no_flight_radius * 2.0 + safe_margin
    activate_dist = threshold * activate
    release_dist = threshold * release
    x1, y1 = own_pos
    committed = anchor is not None

    rvx = rvy = 0.0
    for other in others:
        dx = x1 - other[0]
        dy = y1 - other[1]
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            continue
        if not committed and dist >= activate_dist:
            continue
        mag = _repulsion_magnitude(dist, release_dist, gain)
        if mag <= 0.0:
            continue
        rvx += dx / dist * mag
        rvy += dy / dist * mag

    total = math.hypot(rvx, rvy)
    if total < 1e-6:
        return cmd, None

    desired = (rvx / total, rvy / total)
    if anchor is not None and (anchor[0] * desired[0]
                               + anchor[1] * desired[1]) > 0.15:
        direction = anchor          # hold the committed escape side
    else:
        direction = desired         # genuine change (e.g. after passing)
    cmd.linear.x = float(direction[0] * total)
    cmd.linear.y = float(direction[1] * total)
    return cmd, direction


def method_1_n_drones(own_pos, others, **kwargs):
    """Sum of method_1 repulsion from every other drone (N-generic)."""
    cmd, _ = method_1_field(own_pos, others, **kwargs)
    return cmd


def _as_position_list(others):
    """Normalise ``others`` to a list of 3-tuples.

    Accepts the old 2-drone call style — a single (x, y, z) tuple — and the
    N-drone style — a sequence of positions.
    """
    if isinstance(others, (tuple, list)) and others and \
            isinstance(others[0], (int, float)):
        return [tuple(others)]
    return [tuple(o) for o in others]


def method_2_3d_swap(own_pos, others, own_goal=None, params=None):
    """3D-SWAP: full 3D reference velocity for the ego drone (method_2).

    Runs the 3D-SWAP state machine for the ego drone. ``others`` may be a
    single (x, y, z) tuple or a sequence of them — every other drone behaves as
    a dynamic obstacle in the COD. Unlike method_1, the result is a complete
    reference velocity (horizontal + vertical), not an offset: the drone
    controllers forward it unchanged while this method is active.

    own_pos / others / own_goal are 3-tuples (x, y, z) in the world-aligned
    metre frame; own_goal is the drone's own waypoint. The LIDAR/external-
    obstacle part of the paper is skipped by design.
    """
    params = params or swap_3d.SWAP3DParams()
    res = swap_3d.compute_reference_velocity(own_pos, own_goal or (0, 0, 0),
                                             _as_position_list(others), params)
    cmd = Twist()
    cmd.linear.x = float(res.vx)
    cmd.linear.y = float(res.vy)
    cmd.linear.z = float(res.vz)
    return cmd


def _twist_from(vx, vy, vz):
    cmd = Twist()
    cmd.linear.x = float(vx)
    cmd.linear.y = float(vy)
    cmd.linear.z = float(vz)
    return cmd


def method_3_dpso(own_pos, others, own_goal=None, params=None, state=None,
                  others_vel=None, external_gbests=None):
    """DPSO receding-horizon reference velocity (method_3).

    Wraps dpso_swarm.method_3_dpso and returns ``(Twist, new_state)`` so the
    node can keep the per-drone horizon alive across ticks.
    """
    own_goal = own_goal or (0, 0, 0)
    vx, vy, vz, new_state = dpso_swarm.method_3_dpso(
        own_pos, others, own_goal, params, state,
        others_vel=others_vel, external_gbests=external_gbests)
    return _twist_from(vx, vy, vz), new_state


def method_4_convex_hull(own_pos, others, own_goal=None, params=None,
                         state=None, others_vel=None, external_gbests=None):
    """Convex-hull decongestion reference velocity (method_4)."""
    own_goal = own_goal or (0, 0, 0)
    vx, vy, vz = convex_hull_avoid.method_4_convex_hull(
        own_pos, others, own_goal, params)
    return _twist_from(vx, vy, vz), None


def method_5_gmm(own_pos, others, own_goal=None, params=None, state=None,
                 others_vel=None, external_gbests=None):
    """GMM density-avoidance reference velocity (method_5)."""
    own_goal = own_goal or (0, 0, 0)
    vx, vy, vz = gmm_avoid.method_5_gmm(own_pos, others, own_goal, params)
    return _twist_from(vx, vy, vz), None


def method_6_combined(own_pos, others, own_goal=None, params=None, state=None,
                      others_vel=None, external_gbests=None):
    """Combined GMM + convex-hull + DPSO reference velocity (method_6)."""
    own_goal = own_goal or (0, 0, 0)
    vx, vy, vz, new_state = combined_avoid.method_6_combined(
        own_pos, others, own_goal, params, state,
        others_vel=others_vel, external_gbests=external_gbests)
    return _twist_from(vx, vy, vz), new_state


METHODS = {
    'method_1': method_1_no_flight_zone,
    'method_2': method_2_3d_swap,
    'method_3': method_3_dpso,
    'method_4': method_4_convex_hull,
    'method_5': method_5_gmm,
    'method_6': method_6_combined,
}

# Full-reference methods replace the toward-goal drive while active (the node
# publishes their Twist every tick, like method_2 already does).
FULL_REFERENCE = frozenset(
    ['method_2', 'method_3', 'method_4', 'method_5', 'method_6'])


class CollisionAvoidanceNode(Node):
    def __init__(self):
        super().__init__('drone_collision_avoidance')
        self.method_name = self.declare_parameter(
            'collision_avoidance_method', 'method_1').get_parameter_value().string_value

        if self.method_name not in METHODS:
            self.get_logger().warn(
                f'Unknown method "{self.method_name}", falling back to method_1.')
            self.method_name = 'method_1'
        self.avoidance_fn = METHODS[self.method_name]

        # N-drone topic prefixes and per-drone targets (flattened 3*n).
        # Backward-compatible defaults reproduce the old drone_1/drone_2 pair.
        self.prefixes = list(self.declare_parameter(
            'drone_prefixes', ['drone_1', 'drone_2'])
            .get_parameter_value().string_array_value)
        flat_targets = list(self.declare_parameter(
            'targets', [2.0, 5.0, 1.0, -2.0, 5.0, 1.0])
            .get_parameter_value().double_array_value)

        n = len(self.prefixes)
        if len(flat_targets) != 3 * n:
            self.get_logger().warn(
                f'Expected {3 * n} target values for {n} drones, got '
                f'{len(flat_targets)}; falling back to zero targets.')
            flat_targets = [0.0, 0.0, 0.0] * n
        self.goals = {
            prefix: tuple(flat_targets[3 * i:3 * i + 3])
            for i, prefix in enumerate(self.prefixes)
        }

        self.pos = {}  # prefix -> (x, y, z) world metres
        self.subs = []
        for i, prefix in enumerate(self.prefixes):
            self.subs.append(self.create_subscription(
                NavSatFix, f'/{prefix}/gps',
                lambda msg, p=prefix: self._gps_cb(p, msg), 10))
        self.pubs = {
            prefix: self.create_publisher(Twist, f'/{prefix}/cmd_vel', 10)
            for prefix in self.prefixes
        }

        self.ref_lat = 50.0
        self.ref_lon = 19.0

        # 3D-SWAP geometry, overridable at launch time.
        self.swap_params = swap_3d.SWAP3DParams(
            rc=self.declare_parameter('swap_rc', swap_3d.SWAP3DParams.rc)
            .get_parameter_value().double_value,
            rr=self.declare_parameter('swap_rr', swap_3d.SWAP3DParams.rr)
            .get_parameter_value().double_value,
            hr=self.declare_parameter('swap_hr', swap_3d.SWAP3DParams.hr)
            .get_parameter_value().double_value,
            hb=self.declare_parameter('swap_hb', swap_3d.SWAP3DParams.hb)
            .get_parameter_value().double_value,
            va=self.declare_parameter('swap_va', swap_3d.SWAP3DParams.va)
            .get_parameter_value().double_value,
            vmax=self.declare_parameter('swap_vmax', swap_3d.SWAP3DParams.vmax)
            .get_parameter_value().double_value,
        )

        # Committed escape direction for method_1 hysteresis (set per drone).
        self._anchor = {pref: None for pref in self.prefixes}

        # method_1 goal capture: inside this distance from its own waypoint a
        # drone fades its avoidance authority out so it can finish the landing.
        self.method1_capture = self.declare_parameter(
            'method1_goal_capture', 1.5).get_parameter_value().double_value

        # method_3/6: DPSO trajectory planner tunables. The launch defaults are
        # tuned for the 2-drone pair; for N-drone sessions the per-drone search
        # budget is scaled down so the single 20 Hz callback can still serve
        # every drone (pop*max_iter ~ N^-1). N <= 2 keeps the exact tuned
        # values; the goal-seed warm start keeps the lighter searches robust.
        dp = dpso_swarm.DPSOParams
        n = len(self.prefixes)
        base_pop = int(self.declare_parameter('dpso_pop', dp.pop)
                       .get_parameter_value().integer_value)
        base_max_iter = int(self.declare_parameter(
            'dpso_max_iter', dp.max_iter).get_parameter_value().integer_value)
        if n > 2:
            eff_pop = max(6, round(base_pop * 2.0 / n))
            eff_max_iter = max(6, round(base_max_iter * 2.0 / n))
            self.get_logger().info(
                f'DPSO CPU budget scaled for {n} drones: pop {base_pop} -> '
                f'{eff_pop}, max_iter {base_max_iter} -> {eff_max_iter}.')
        else:
            eff_pop, eff_max_iter = base_pop, base_max_iter
        self.dpso_params = dpso_swarm.DPSOParams(
            pop=eff_pop,
            max_iter=eff_max_iter,
            horizon=int(self.declare_parameter('dpso_horizon', dp.horizon)
                        .get_parameter_value().integer_value),
            plan_dt=self.declare_parameter('dpso_plan_dt', dp.plan_dt)
                .get_parameter_value().double_value,
            replan_ticks=int(self.declare_parameter(
                'dpso_replan_ticks', dp.replan_ticks)
                .get_parameter_value().integer_value),
            w=self.declare_parameter('dpso_w', dp.w)
                .get_parameter_value().double_value,
            c1=self.declare_parameter('dpso_c1', dp.c1)
                .get_parameter_value().double_value,
            c2=self.declare_parameter('dpso_c2', dp.c2)
                .get_parameter_value().double_value,
            w_goal=self.declare_parameter('dpso_w_goal', dp.w_goal)
                .get_parameter_value().double_value,
            w_ob=self.declare_parameter('dpso_w_ob', dp.w_ob)
                .get_parameter_value().double_value,
            w_effort=self.declare_parameter('dpso_w_effort', dp.w_effort)
                .get_parameter_value().double_value,
            w_smooth=self.declare_parameter('dpso_w_smooth', dp.w_smooth)
                .get_parameter_value().double_value,
            keepout=self.declare_parameter('dpso_keepout', dp.keepout)
                .get_parameter_value().double_value,
            ob_stop=self.declare_parameter('dpso_ob_stop', dp.ob_stop)
                .get_parameter_value().double_value,
            speed_cap=self.declare_parameter('dpso_speed_cap', dp.speed_cap)
                .get_parameter_value().double_value,
        )

        # method_4: convex-hull tunables.
        hp = convex_hull_avoid.HullParams
        self.hull_params = convex_hull_avoid.HullParams(
            activation=self.declare_parameter('hull_activation', hp.activation)
                .get_parameter_value().double_value,
            gain=self.declare_parameter('hull_gain', hp.gain)
                .get_parameter_value().double_value,
            base_speed=self.declare_parameter('hull_base_speed', hp.base_speed)
                .get_parameter_value().double_value,
            speed_cap=self.declare_parameter('hull_speed_cap', hp.speed_cap)
                .get_parameter_value().double_value,
        )

        # method_5: GMM tunables.
        gp = gmm_avoid.GMMParams
        self.gmm_params = gmm_avoid.GMMParams(
            sigma=self.declare_parameter('gmm_sigma', gp.sigma)
                .get_parameter_value().double_value,
            gain=self.declare_parameter('gmm_gain', gp.gain)
                .get_parameter_value().double_value,
            activation=self.declare_parameter('gmm_activation', gp.activation)
                .get_parameter_value().double_value,
            base_speed=self.declare_parameter('gmm_base_speed', gp.base_speed)
                .get_parameter_value().double_value,
            speed_cap=self.declare_parameter('gmm_speed_cap', gp.speed_cap)
                .get_parameter_value().double_value,
        )

        # method_6: combined tunables (reuse the same dpso_params + extras).
        cp = combined_avoid.CombinedParams
        self.combined_params = combined_avoid.CombinedParams(
            dpso=self.dpso_params,
            gmm_sigma=self.gmm_params.sigma,
            comb_w_gmm=self.declare_parameter('comb_w_gmm', cp.comb_w_gmm)
                .get_parameter_value().double_value,
            stgo_adoption=self.declare_parameter(
                'stgo_adoption', cp.stgo_adoption)
                .get_parameter_value().double_value,
        )

        # Persistent state for the stateful methods (receding horizon/gbest).
        self.tick_dt = self.dpso_params.tick_dt
        self._state = {pref: None for pref in self.prefixes}
        self._prev_pos = {}  # prefix -> (x, y, z) from the previous tick
        # Per-drone accumulated flight distance (m), reported in the perf JSON.
        self._path_m = {pref: 0.0 for pref in self.prefixes}

        # Airframe-aware authority per drone (aligned with drone_prefixes).
        # Heavy airframes are commanded at fractionally gentler avoidance
        # speeds (see scale_authority). Defaults to full authority so a bare
        # node invocation reproduces the classic behaviour.
        self.authorities = list(self.declare_parameter(
            'mass_authorities', [1.0] * n)
            .get_parameter_value().double_array_value)
        if len(self.authorities) != n:
            self.get_logger().warn(
                f'Expected {n} mass authorities for {n} drones, got '
                f'{len(self.authorities)}; falling back to full authority.')
            self.authorities = [1.0] * n

        self.timer = self.create_timer(0.05, self.check_and_avoid)
        self.get_logger().info(
            f'Collision avoidance active with method: {self.method_name} '
            f'for {n} drones: {", ".join(self.prefixes)}')

        # Optional per-tick performance probe (default off, only for the
        # method-comparison benchmark).  When enabled, every tick's wall/CPU
        # time and achieved interval are appended to
        # ``<perf_dir>/perf_<method>.csv`` (kill-safe) and summarised to
        # ``.../perf_<method>.json`` on shutdown.
        self.perf_probe = self.declare_parameter(
            'perf_probe', False).get_parameter_value().bool_value
        self.perf_dir = self.declare_parameter(
            'perf_dir', '').get_parameter_value().string_value
        self._perf_fh = None
        self._perf_wall = []
        self._perf_cpu = []
        self._perf_dt = []
        self._last_tick_s = None
        if self.perf_probe and self.perf_dir:
            try:
                os.makedirs(self.perf_dir, exist_ok=True)
                path = os.path.join(self.perf_dir,
                                    f'perf_{self.method_name}.csv')
                self._perf_fh = open(path, 'w', buffering=1)
                self._perf_fh.write('tick_wall_ns,cpu_ns,interval_s\n')
            except Exception as exc:  # noqa: BLE001
                self._perf_fh = None
                self.get_logger().warn(
                    'perf_probe enabled but csv setup failed - timing won\'t '
                    f'be saved: {exc!r}\n{traceback.format_exc()}')

    def _record_perf(self, wall_ns, cpu_ns, dt_s):
        self._perf_wall.append(wall_ns)
        self._perf_cpu.append(cpu_ns)
        self._perf_dt.append(dt_s)
        if self._perf_fh is not None:
            self._perf_fh.write(f'{wall_ns},{cpu_ns},{dt_s:.9f}\n')

    def _dump_perf(self):
        """Best-effort summary JSON + close of the per-tick CSV probe."""
        fh, self._perf_fh = self._perf_fh, None
        self.get_logger().info(
            f'perf dump called with n={len(self._perf_wall)} '
            f'closing_fd={fh.fileno() if fh else None}')
        if fh is not None:
            try:
                fh.close()
            except Exception:  # noqa: BLE001
                pass
        if not self._perf_wall:
            return
        try:
            summary = {
                'method': self.method_name,
                'n': len(self._perf_wall),
                'wall_ms': {
                    'mean': statistics.mean(self._perf_wall) / 1e6,
                    'p50': _pct(self._perf_wall, 50) / 1e6,
                    'p95': _pct(self._perf_wall, 95) / 1e6,
                    'max': max(self._perf_wall) / 1e6,
                },
                'cpu_ms': {
                    'mean': statistics.mean(self._perf_cpu) / 1e6,
                    'p50': _pct(self._perf_cpu, 50) / 1e6,
                    'p95': _pct(self._perf_cpu, 95) / 1e6,
                    'max': max(self._perf_cpu) / 1e6,
                },
                'interval_s': {
                    'mean': statistics.mean(self._perf_dt),
                    'p50': _pct(self._perf_dt, 50),
                    'max': max(self._perf_dt) if self._perf_dt else None,
                },
                'rate_hz': {
                    'mean': 1.0 / statistics.mean(self._perf_dt)
                    if self._perf_dt else None,
                },
                'path_m': dict(self._path_m),
                'path_max': max(self._path_m.values())
                if self._path_m else None,
            }
        except Exception:  # noqa: BLE001
            self.get_logger().warn('perf summary computation failed.')
            return
        try:
            path = os.path.join(self.perf_dir,
                                f'perf_{self.method_name}.json')
            with open(path, 'w') as f:
                json.dump(summary, f, indent=2)
            self.get_logger().info(
                f'Perf probe: method={self.method_name} '
                f'wall p50={summary["wall_ms"]["p50"]:.3f} ms '
                f'p95={summary["wall_ms"]["p95"]:.3f} ms '
                f'rate={summary["rate_hz"]["mean"]:.1f} Hz '
                f'(n={summary["n"]}).')
        except Exception:  # noqa: BLE001
            self.get_logger().warn('perf summary not saved.')

    def _gps_cb(self, prefix, msg):
        self.pos[prefix] = (msg.latitude, msg.longitude, msg.altitude)

    def gps_to_xy(self, prefix, lat, lon):
        x = (lat - self.ref_lat) * 111320.0
        y = (lon - self.ref_lon) * 111320.0 * math.cos(math.radians(self.ref_lat))
        return x, y

    def _world_pos(self, prefix):
        p = self.pos.get(prefix)
        if p is None:
            return None
        x, y = self.gps_to_xy(prefix, p[0], p[1])
        return (x, y, p[2])

    def check_and_avoid(self):
        if self.perf_probe:
            t0 = time.perf_counter_ns()
            c0 = time.process_time_ns()
        try:
            self._check_and_avoid()
        except Exception:  # noqa: BLE001
            self.get_logger().error(
                'Avoidance callback raised, skipping tick:\n'
                + traceback.format_exc())
            return
        if self.perf_probe:
            now_s = time.perf_counter()
            dt = now_s - self._last_tick_s if self._last_tick_s is not None \
                else 0.0
            self._last_tick_s = now_s
            self._record_perf(time.perf_counter_ns() - t0,
                              time.process_time_ns() - c0, dt)

    def _check_and_avoid(self):
        positions = {}
        for pref in self.prefixes:
            p = self.pos.get(pref)
            if p is None:
                return
            positions[pref] = self._world_pos(pref)

        # Per-tick neighbour velocity estimate (for DPSO lane prediction).
        vel_map = {}
        for pref in self.prefixes:
            prev = self._prev_pos.get(pref)
            if prev is not None:
                d = tuple(positions[pref][k] - prev[k] for k in range(3))
                self._path_m[pref] = self._path_m.get(pref, 0.0) + \
                    math.sqrt(sum(d[k] * d[k] for k in range(3)))
                vel_map[pref] = tuple(d[k] / self.tick_dt for k in range(3))
            else:
                vel_map[pref] = (0.0, 0.0, 0.0)
        self._prev_pos = {pref: positions[pref] for pref in self.prefixes}

        if self.method_name == 'method_1':
            for i, pref in enumerate(self.prefixes):
                own_xy = positions[pref][:2]
                others_xy = [positions[o][:2] for o in self.prefixes
                             if o != pref]
                cmd, self._anchor[pref] = method_1_field(
                    own_xy, others_xy, self._anchor[pref])
                gx, gy = self.goals[pref][:2]
                gd = math.hypot(own_xy[0] - gx, own_xy[1] - gy)
                weight = max(0.0, min(1.0, (gd - 0.3)
                                      / max(self.method1_capture - 0.3, 1e-6)))
                cmd.linear.x *= weight
                cmd.linear.y *= weight
                cmd = scale_authority(cmd, self.authorities[i])
                if cmd.linear.x or cmd.linear.y:
                    self.get_logger().warn(
                        f'No-flight-zone overlap, {pref} push='
                        f'({cmd.linear.x:.3f}, {cmd.linear.y:.3f})',
                        throttle_duration_sec=0.5)
                self.pubs[pref].publish(cmd)
            return

        if self.method_name not in FULL_REFERENCE:
            # Unknown / legacy fallback — command nothing.
            for pref in self.prefixes:
                self.pubs[pref].publish(Twist())
            return

        others = {pref: [positions[o] for o in self.prefixes if o != pref]
                  for pref in self.prefixes}
        others_vel = {pref: [vel_map[o] for o in self.prefixes if o != pref]
                      for pref in self.prefixes}

        # gbest exchange (paper: receive P_{j,gbest} / broadcast P_{i,gbest}):
        # feed each stateful drone the gbest trajectories computed for its
        # neighbours on the *previous* tick.
        gbests = {}

        if self.method_name == 'method_2':
            for i, pref in enumerate(self.prefixes):
                cmd = method_2_3d_swap(positions[pref], others[pref],
                                       self.goals[pref], self.swap_params)
                cmd = scale_authority(cmd, self.authorities[i])
                r = swap_3d.compute_reference_velocity(
                    positions[pref], self.goals[pref], others[pref],
                    self.swap_params)
                phi_txt = f'{r.phi_a:.2f}' if r.phi_a is not None else '--'
                self.get_logger().info(
                    f'3D-SWAP {pref}: xy={r.s_xy} z={r.s_z} '
                    f'v=({cmd.linear.x:.2f},{cmd.linear.y:.2f},'
                    f'{cmd.linear.z:.2f}) phi_a={phi_txt}',
                    throttle_duration_sec=0.5,
                )
                self.pubs[pref].publish(cmd)
            return

        if self.method_name == 'method_6':
            # --- STGO two-pass: local DPSO -> global consensus -> prep ---
            # Pass 1: every drone runs its local DPSO with GMM extra_cost.
            # Its external references: the *previous* tick's adopted plan
            # (owned by the winner) at the STGO keep-out weight, plus each
            # other drone's own recent gbest at the default blend.  The
            # winner never avoids its own adopted plan.
            vels = {}
            local_plans = {}
            for i, pref in enumerate(self.prefixes):
                params = self._method_params(self.method_name)
                ext = []
                wts = []
                for o in self.prefixes:
                    if o == pref:
                        continue
                    st_o = self._state.get(o)
                    cons = None
                    owner = None
                    if st_o is not None:
                        cons = getattr(st_o, 'consensus_gbest', None)
                        owner = getattr(st_o, 'consensus_owner', None)
                    if cons is not None and owner == o:
                        ext.append(cons)
                        wts.append(params.stgo_adoption)
                    elif st_o is not None and getattr(st_o, 'gbest', None) \
                            is not None:
                        ext.append(st_o.gbest)
                        wts.append(None)
                    else:
                        ext.append(None)
                        wts.append(None)
                vx, vy, vz, st = combined_avoid.method_6_combined(
                    positions[pref], others[pref], self.goals[pref], params,
                    self._state[pref], others_vel=others_vel[pref],
                    external_gbests=ext if any(ext) else None,
                    external_weights=wts if any(w is not None for w in wts)
                    else None, self_id=pref)
                self._state[pref] = st
                vels[pref] = (vx, vy, vz)
                if st is not None and getattr(st, 'gbest', None) is not None:
                    local_plans[pref] = (st.gbest, st.gbest_cost)

            # STGO consensus: select the globally cheapest plan.
            winner = combined_avoid.stgo_consensus(local_plans)
            adopted = (local_plans.get(winner, (None, 0.0))[0]
                       if winner is not None else None)

            # Pass 2: remember the adopted plan as the owner-tagged consensus
            # reference (each drone still drives its own local plan this tick).
            for i, pref in enumerate(self.prefixes):
                params = self._method_params(self.method_name)
                self._state[pref] = combined_avoid.method_6_prepare(
                    self._state[pref], adopted, params,
                    adoption_weight=params.stgo_adoption, owner=winner)
                cmd = _twist_from(*vels[pref])
                cmd = scale_authority(cmd, self.authorities[i])
                if cmd.linear.x or cmd.linear.y or cmd.linear.z:
                    self.get_logger().info(
                        f'{self.method_name} {pref}: '
                        f'v=({cmd.linear.x:.2f},{cmd.linear.y:.2f},'
                        f'{cmd.linear.z:.2f}) stgo={winner}',
                        throttle_duration_sec=0.5)
                self.pubs[pref].publish(cmd)
        else:
            # --- Single-pass for methods 3/4/5 ---
            for i, pref in enumerate(self.prefixes):
                fn = METHODS[self.method_name]
                params = self._method_params(self.method_name)
                ext = [gbests[o] for o in self.prefixes if o != pref
                       and gbests.get(o) is not None]
                cmd, self._state[pref] = fn(
                    positions[pref], others[pref], self.goals[pref], params,
                    self._state[pref], others_vel=others_vel[pref],
                    external_gbests=ext)
                cmd = scale_authority(cmd, self.authorities[i])
                if cmd.linear.x or cmd.linear.y or cmd.linear.z:
                    self.get_logger().info(
                        f'{self.method_name} {pref}: '
                        f'v=({cmd.linear.x:.2f},{cmd.linear.y:.2f},'
                        f'{cmd.linear.z:.2f})',
                        throttle_duration_sec=0.5)
                self.pubs[pref].publish(cmd)
                st = self._state[pref]
                if st is not None and getattr(st, 'gbest', None) is not None:
                    gbests[pref] = st.gbest

    def _method_params(self, method_name):
        if method_name == 'method_3':
            return self.dpso_params
        if method_name == 'method_4':
            return self.hull_params
        if method_name == 'method_5':
            return self.gmm_params
        if method_name == 'method_6':
            return self.combined_params
        return None


def main(args=None):
    rclpy.init(args=args)
    node = CollisionAvoidanceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._dump_perf()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()