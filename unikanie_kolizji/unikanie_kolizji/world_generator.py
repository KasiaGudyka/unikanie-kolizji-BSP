#!/usr/bin/env python3

"""Generate an N-drone Gazebo Harmonic world plus its ROS<->GZ bridge config.

Pattern follows the ``bebop_gz/world_generator.py`` used by the swarmbebop
launch: a Python generator produces a world whose model count changes with the
session. Unlike swarmbebop (which rewrites one file in place), this generator
emits a *fresh* world into a writable output directory; the launch file passes
gazebo a unique ``-z <seed>`` so every session runs in a transient world and
stale-state glitches (double/spurious drones) cannot recur.

Layout used by every generated world:
  * N drones ``drone_0 .. drone_{N-1}`` starting on the y=0 line (x symmetric
    about 0, 2 m spacing) and flying to mirrored ``x`` targets on y=5, so
    opposite drones cross paths in the middle — the collision-avoidance
    scenario.
  * N destination spheres ``target_i`` (visual-only, one per drone).
  * Physics automatically relaxed when N >= 5 (the multicopter models are
    solver-heavy) — lower update rate, larger step, fewer solver iterations.
  * Optional ``gz-sim-wind-effects-system`` plugin plus a world ``<wind>``
    element whenever the wind setting is not 'off'. Each option encodes the
    mode and speed, e.g. 'strong_5' (constant 5 m/s) or 'gust_3' (3 m/s with
    a sinusoidal gust). The WindEffects plugin only *generates the wind field*
    (its own force is zeroed via <force_approximation_scaling_factor>0</...>);
    per-drone the native ``gz_aero_drag`` plugin applies the real quadratic
    drag law F = 0.5 * rho * Cd * A * |v_rel| * v_rel, so a 20 kg field quad
    barely buffers in a 5 m/s wind while the 0.5 kg toy is pushed.
"""

import argparse
import glob
import math
import os
import shutil
import textwrap

# Drone masses the compat layer offers (kg). The 0.5 kg entry is the real
# Parrot Bebop 2 model (shipped model.sdf, body mass BODY_MASS_KG below).
# Heavier sizes spawn a *spec-driven rewrite* of the same geometry carrying
# the physics of a real commercial drone of that weight class (see
# AIRFRAME_SPECS) instead of a toy rescale.
DRONE_MASS_OPTIONS = (0.5, 5.0, 20.0)

# Exact body mass of the shipped parrot_bebop_2 model.sdf (kg): the physics
# the pristine 0.5 kg option loads.
BODY_MASS_KG = 0.503

# Rotor thrust / torque coefficients of the shipped model (rotor force =
# k * w^2, rotor torque = momentK * w^2). These calibrate the real Bebop 2:
# hovering it holds ~1.23 N per rotor at ~3600 RPM, which matches the real
# toy. Heavy airframes carry their own k (hover / motor-headroom derived) and
# a momentConstant scaled by the same factor so the torque-to-thrust ratio
# (yaw authority per newton) stays physically consistent across sizes.
ROTOR_K = 8.54858e-06
ROTOR_MOMENT_K = 0.005964552

# Nominal rotor-arm radius of the shipped model (m). Each propeller link sits
# at |r| ~ 0.1455 m from the body frame. Heavier airframes rewrite the four
# propeller poses radially outward to this reference x (arm) scale so the
# rotor geometry spans the real wheelbase.
PROP_ARM_REF_M = 0.1455

# Propeller link names, in the SDF link declaration order of model.sdf.
PROP_LINK_NAMES = ('propeller_rr', 'propeller_rl', 'propeller_fr',
                   'propeller_fl')

# Real-airframe physics per drone mass (kg). rotor_k and max_rot_velocity come
# from real-aircraft hover/motor numbers; the body inertia is the real
# airframe's (approximated, since gz-sim ships no heavy multicopter model):
#   5.0 kg  -> DJI Matrice 300 RTK class quad: diagonal wheelbase 895 mm,
#              ~21" props, MTOW 9.0 kg (flies ~5 kg armed). Per-rotor hover
#              = 5g/4 = 12.3 N at k = 5.8e-05 (~460 rad/s = 4390 RPM), TWR
#              ~2.2 => max ~682 rad/s = ~6500 RPM (real 21" motor range).
#   20.0 kg -> DJI Agras T10 class field quad*: max rotor distance 1480 mm,
#              33" props, motor max pull ~11.2 kgf (~110 N). Per-rotor hover
#              = 20g/4 = 49 N at k = 8.2e-04 (~244 rad/s = 2335 RPM), TWR
#              ~2.2 => max ~363 rad/s = ~3465 RPM. (* T10 is a hexacopter; a
#              4-rotor 20 kg field drone is also a real class, e.g. XAG P-series.)
# inertia is the body tensor (ixx, ixy, ixz, iyy, iyz, izz) in kg.m^2.
# max_rot_velocity is in rad/s, like the shipped Bebop's 800.0 cap.
# 0.5 kg references -> untouched shipped model (no copy is written).
# Each spec also carries the exposed cross-sectional wind area (m^2) and drag
# coefficient used by the native gz_aero_drag plugin: the real dynamic-pressure
# law F = 0.5 * rho * Cd * A * |v_rel| * v_rel is area-based and independent of
# body mass, so a 20 kg field quad barely buffers in 5 m/s while the 0.5 kg toy
# is pushed (see also option_compatibility.WIND_DOC).
AIRFRAME_SPECS = {
    0.5: {'name': 'Parrot Bebop 2 (toy reference)',
          'mass_kg': BODY_MASS_KG,
          'arm_m': PROP_ARM_REF_M,
          'rotor_k': ROTOR_K,
          'max_rot_velocity': 800.0,
          'inertia': None,
          'hover_rpm': None,
          'drag_area_m2': 0.05,
          'drag_cd': 1.0},
    5.0: {'name': 'DJI Matrice 300 RTK class quad (5 kg)',
          'mass_kg': 5.0,
          'arm_m': 0.4475,
          'rotor_k': 5.8e-05,
          'max_rot_velocity': 682.0,
          'inertia': (0.42, 0.0, 0.0, 0.40, 0.0, 0.62),
          'hover_rpm': None,
          'drag_area_m2': 0.15,
          'drag_cd': 1.0},
    20.0: {'name': 'DJI Agras T10 class field quad (20 kg)',
           'mass_kg': 20.0,
           'arm_m': 0.740,
           'rotor_k': 8.2e-04,
           'max_rot_velocity': 362.6,
           'inertia': (7.2, 0.0, 0.0, 6.8, 0.0, 12.0),
           'hover_rpm': None,
           'drag_area_m2': 0.30,
           'drag_cd': 1.0},
}


def hover_rpm(spec: dict) -> float:
    """Hover rotor speed (RPM) of a 4-rotor airframe spec."""
    omega = math.sqrt(spec['mass_kg'] * 9.80665 / (4.0 * spec['rotor_k']))
    return omega * 60.0 / (2.0 * math.pi)

# Per-mass MulticopterVelocityControl tuning. The plugin divides
# attitudeGain / angularRateGain by the vehicle inertia (they are specified
# for a unit-inertia airframe), so with a real heavy airframe's much larger
# inertia the same gains yield a far weaker attitude loop and the drone cannot
# hold level / climb. Every mass therefore scales attitude & rate gains by the
# inertia ratio (real / toy, per axis) and a sedate factor so they fly
# controllably but responsibly; the velocity loop is kept authoritative enough
# to reject disturbances and linear acceleration / velocity caps keep a heavy
# body from over-commanding and overshooting. The 0.5 kg entry scales the same
# way from the real Bebop inertia tensor (Ixx 1.805e-3, Iyy 1.764e-3,
# Izz 3.328e-3 kg.m^2) - previously it emitted the toy's bare gains untouched,
# and with the real (550x smaller) inertia the un-scaled inner loop was
# roughly 4x stiffer than the validated heavy scheme, which made the small
# quad tumble (pitch over) on setpoint steps in calm air. Caps for 0.5 kg cap
# the plugin's default ~25 m/s^2 authority at a Bebop-like envelope.
# Wind-mode worlds swap in the elevated WIND_AUTHORITY caps (methods stay
# identical) so the quad can hold against a real wind.
MASS_CONTROL = {
    0.5: {'velocity_gain': '2 2.2 2',
          'attitude_gain': '0.2 0.2 0.2',
          'angular_rate_gain': '0.03 0.03 0.05',
          'max_linear_accel': '',
          'max_linear_vel': ''},
    5.0: {'velocity_gain': '1.8 1.9 1.8',
          'attitude_gain': '11.6 11.3 9.31',
          'angular_rate_gain': '1.75 1.70 2.33',
          'max_linear_accel': '1.5 1.5 1.8',
          'max_linear_vel': '1.5 1.5 1.5'},
    20.0: {'velocity_gain': '1.8 1.9 1.8',
           'attitude_gain': '199 193 180',
           'angular_rate_gain': '29.9 28.9 45.1',
           'max_linear_accel': '1.2 1.2 1.4',
           'max_linear_vel': '1.3 1.3 1.3'},
}

# Wind-mode control authority: the same velocity/attitude/rate gains as above
# (stability-critical), but with linear caps raised far above the sedate cruise
# so a real-airframe quad can command 4-6 m/s to hold against a blowing wind.
# Used ONLY when wind != 'off'; calm worlds emit the exact validated values.
WIND_AUTHORITY = {
    0.5: {'max_linear_accel': '4.0 4.0 4.0',
          'max_linear_vel': '4.0 4.0 4.0'},
    5.0: {'max_linear_accel': '4.0 4.0 4.0',
          'max_linear_vel': '6.0 6.0 6.0'},
    20.0: {'max_linear_accel': '3.0 3.0 3.0',
           'max_linear_vel': '5.0 5.0 5.0'},
}


def _velocity_control_extra(drone_mass_kg: float, wind: str = 'off') -> str:
    """Extra <velocityGain>/<maximum...>/attitude elements for a mass.

    identity for the 0.5 kg baseline. ``wind != 'off'`` swaps in the elevated
    WIND_AUTHORITY caps so the quad can fly into a blowing wind.
    """
    cfg = MASS_CONTROL.get(drone_mass_kg, MASS_CONTROL[0.5])
    if wind != 'off':
        wa = WIND_AUTHORITY[drone_mass_kg]
        cfg = {**cfg, **wa}
    lines = [f'<velocityGain>{cfg["velocity_gain"]}</velocityGain>']
    if cfg['max_linear_accel']:
        lines.append(f'<maximumLinearAcceleration>'
                     f'{cfg["max_linear_accel"]}</maximumLinearAcceleration>')
    if cfg['max_linear_vel']:
        lines.append(f'<maximumLinearVelocity>'
                     f'{cfg["max_linear_vel"]}</maximumLinearVelocity>')
    lines.append(f'<attitudeGain>{cfg["attitude_gain"]}</attitudeGain>')
    lines.append(f'<angularRateGain>{cfg["angular_rate_gain"]}'
                 f'</angularRateGain>')
    return '\n'.join(lines)

# Wind settings, matching option_compatibility.WIND_OPTIONS.
# Every non-'off' value encodes mode + speed (m/s):
#   'strong_N'  - uniform N m/s cross-wind along +x (no gusts).
#   'gust_N'    - same N m/s seed but the WindEffects plugin adds a
#                 sinusoidal magnitude gust and directional wobble.
WIND_OPTIONS = (
    'off',
    'strong_1', 'strong_3', 'strong_5', 'strong_10',
    'gust_1', 'gust_3', 'gust_5', 'gust_10',
)

# Wind speed (m/s) for each option. Source of truth for the world <wind> seed;
# option_compatibility.WIND_SPEED_MS mirrors it (asserted in the tests).
WIND_SPEED_MS: dict = {w: 0.0 if w == 'off' else float(w.split('_', 1)[1])
                       for w in WIND_OPTIONS}

# Axis the wind blows along (+x = cross-wind to the y=0 -> y=5 flight lane,
# so every drone is pushed sideways on the same axis and the disturbance is
# easy to observe in x/y pose samples).
WIND_AXIS = 'x'

# Default world base name. gz-sim 8 keeps this exact name even when a -z
# transient seed is passed, so the controller/avoidance POSE_TOPIC stays valid.
WORLD_NAME = 'empty_world_harmonic'

# Spawn placement: start line on y=0, targets on y=5, symmetric x spacing.
SPAWN_SPACING = 2.0
START_Y = 0.0
START_Z = 0.5
START_Z_WIND = 1.0  # wind worlds spawn drones at cruise height: an airborne
# quad lifts off / holds under wind, while a resting one can be pinned and
# slid by a constant side force before its motors unload the landing gear.
TARGET_Y = 5.0
TARGET_Z = 1.0

# Shield palette (visual only). Reused for the matching target sphere colour,
# for the in-sim text label colour and for the GUI trajectory traces so a
# single drone has one consistent colour everywhere.
COLORS = [
    '0.3 0.3 0.8',  # blue
    '0.8 0.3 0.3',  # red
    '0.2 0.7 0.3',  # green
    '0.9 0.7 0.2',  # yellow
    '0.6 0.2 0.8',  # purple
    '0.1 0.8 0.8',  # cyan
    '0.9 0.4 0.6',  # pink
    '0.6 0.6 0.6',  # grey
    '0.8 0.6 0.3',  # tan
    '0.2 0.4 0.8',  # indigo
]


def color_rgb(index: int) -> tuple:
    """Palette entry ``index`` as an ``(r, g, b)`` float triplet."""
    return tuple(float(v) for v in COLORS[index % len(COLORS)].split())


def color_hex(index: int) -> str:
    """Palette entry ``index`` as a ``#rrggbb`` string (for plotting)."""
    r, g, b = color_rgb(index)
    return '#%02x%02x%02x' % (round(r * 255), round(g * 255), round(b * 255))

# Physics settings: full fidelity for small flocks (<=4: the MulticopterVelocityControl
# inner gains are validated at 1000 Hz / 0.001 s / 100 solver iters; relaxing it
# injects phase lag -> overshoot/circling), relaxed for large ones so the
# real-time solver can keep up with many multicopter models. ``turbo=True``
# (headless tuning runs) lifts the 1:1 real-time lock so the physics advances as
# fast as the wall-clock driver nodes/CPU allow; results are then judged from
# saved stats/tlogs, not live GUI elapsed numbers.
def physics_config(num_drones: int, turbo: bool = False) -> dict:
    if num_drones >= 8:
        base = {'max_step_size': 0.008,
                'real_time_update_rate': 125,
                'solver_iters': 24,
                'degraded': True}
    elif num_drones >= 5:
        base = {'max_step_size': 0.002,
                'real_time_update_rate': 250,
                'solver_iters': 40,
                'degraded': True}
    else:
        base = {'max_step_size': 0.001,
                'real_time_update_rate': 500,
                'solver_iters': 100,
                'degraded': False}
    return {**base, 'real_time_factor': 1000.0 if turbo else 1.0}


def spawn_x(num_drones: int) -> list:
    """Symmetric start x coordinates for ``num_drones`` drones."""
    half = SPAWN_SPACING * (num_drones - 1) / 2.0
    return [-half + i * SPAWN_SPACING for i in range(num_drones)]


def target_positions(start_xs: list, lane_offset: float = 0.0,
                     altitude_lane: float = 0.3) -> list:
    """Mirror the start x to create crossing paths on the y=5 target line.

    ``lane_offset`` (default 0) shifts alternate drones' target x by +/- the
    offset (by index parity), breaking the symmetric layout where every path
    crosses through the exact same point. That degenerate layout is the
    hardest case for any pair-potential avoidance method (5-10 lanes through
    one point); a small offset reproduces realistic offset crossing geometry
    while still being deterministic.

    ``altitude_lane`` (m, default 0.3 for N >= 5, ignored for N < 5) lays each
    drone's target z on an altitude ladder ``START_Z-bounded(TARGET_Z + k*off)``
    so the mirrored flock scans in z as well as x. With every drone aimed at
    the same altitude the N-lane crossing degenerates to a single point and
    the pairwise planners can only resolve it at N <= 2; the altitude ladder
    spreads the crossing vertically (like ATC track levels) without any
    central coordinator -- each drone only needs its own index.
    """
    n = len(start_xs)
    targets = []
    for i, x in enumerate(start_xs):
        if n >= 5 and altitude_lane > 0.0:
            twz = max(START_Z, TARGET_Z + altitude_lane * (i - (n - 1) / 2.0))
        else:
            twz = TARGET_Z
        targets.append((-x + (lane_offset if i % 2 == 0 else -lane_offset),
                        TARGET_Y, twz))
    return targets


def markers_plugin_available() -> bool:
    """True if the gz-sim8 markers system library is installed.

    The in-sim TEXT labels (drone_labels node) only render when this library
    exists on the machine.  Some gz-sim8 installs (e.g. the ros-*-gz-sim-vendor
    package) ship without it; probing here lets the world be generated without
    the dead plugin so the launch has no spurious load errors.
    """
    candidate_dirs = [
        '/usr/lib/x86_64-linux-gnu/gz-sim-8/plugins',
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '.'),
    ]
    render_dir = os.environ.get('GZ_SIM_SYSTEM_PLUGIN_PATH', '')
    if render_dir:
        candidate_dirs.append(render_dir)
    patterns = [
        'libgz-sim8-markers-system*',
        'libignition-gazebo*-markers-system*',
    ]
    for d in candidate_dirs:
        for pat in patterns:
            if glob.glob(os.path.join(d, pat)):
                return True
    return False


def _model_sdf_path() -> str:
    """Absolute path of the shipped parrot_bebop_2 model.sdf.

    Resolved from the source tree first (so unit tests pass without an
    overlay), falling back to the ament share directory (installed layout).
    """
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, os.pardir, 'models', 'parrot_bebop_2', 'model.sdf'),
        os.path.join(here, 'models', 'parrot_bebop_2', 'model.sdf'),
    ]
    try:
        from ament_index_python.packages import get_package_share_directory
        candidates.append(os.path.join(
            get_package_share_directory('unikanie_kolizji'),
            'models', 'parrot_bebop_2', 'model.sdf'))
    except Exception:  # noqa: BLE001 - ament_index unavailable (bare runtime)
        pass
    for path in candidates:
        if os.path.isfile(path):
            return os.path.abspath(path)
    raise FileNotFoundError(
        'could not locate parrot_bebop_2/model.sdf (tried: '
        + ', '.join(candidates) + ')')


def _rewrite_model_sdf(sdf: str, spec: dict) -> str:
    """Return a model SDF rewritten to a real-airframe spec.

    Only physics is rewritten: the body ``<inertial>`` carries the real
    airframe mass + inertia tensor, and the four propeller link poses are
    moved radially outward (uniform arm scale) so the rotor geometry spans
    the real wheelbase. Propellers, sensors, collisions, wind flags and
    meshes are byte-identical to the shipped model.
    """
    import re
    body_match = re.search(r'(<link name="body">.*?<inertial>)(.*?)(</inertial>)',
                           sdf, re.S)
    if not body_match:
        raise ValueError('model.sdf has no body <inertial> block to rewrite')
    prefix, inertial, suffix = body_match.groups()
    tensor_tags = ('ixx', 'ixy', 'ixz', 'iyy', 'iyz', 'izz')

    def _entry(match):
        tag = match.group(1)
        if tag == 'mass':
            return f'<{tag}>{spec["mass_kg"]:g}</{tag}>'
        return f'<{tag}>{spec["inertia"][tensor_tags.index(tag)]:g}</{tag}>'

    new_inertial = re.sub(
        r'<(mass|ixx|ixy|ixz|iyy|iyz|izz)>\s*([-+0-9.eE]+)\s*</\1>',
        _entry, inertial)
    sdf = sdf[:body_match.start()] + prefix + new_inertial \
        + suffix + sdf[body_match.end():]

    arm_scale = spec['arm_m'] / PROP_ARM_REF_M
    for prop in PROP_LINK_NAMES:
        def _pose(match, scale=arm_scale):
            prefix = match.group(1)
            x, y, z = (float(match.group(k)) for k in ('x', 'y', 'z'))
            return prefix + (f'{x * scale:g} {y * scale:g} '
                             f'{z:g} 0 0 0</pose>')
        sdf, _ = re.subn(
            rf'(<link name=[\'"]{prop}[\'"]>\s*<pose>)'
            r'(?P<x>[-+0-9.eE]+) (?P<y>[-+0-9.eE]+) (?P<z>[-+0-9.eE]+) '
            r'0 0 0</pose>',
            lambda m: _pose(m), sdf)
    return sdf


def _rotor_moment_constant(spec: dict) -> float:
    """Rotor torque coefficient keeping the thrust-to-yaw ratio of the spec.

    The shipped model's moment-to-thrust ratio is (momentConstant / rotor_k);
    keeping that ratio across airframe sizes leaves yaw authority proportional
    to the (larger) rotor thrust of the real aircraft.
    """
    return ROTOR_MOMENT_K * (spec['rotor_k'] / ROTOR_K)


def _motor_block(namespace: str, joint: str, turning: str, actuator: int,
                 spec: dict) -> str:
    return textwrap.dedent(f'''\
        <plugin
          filename="gz-sim-multicopter-motor-model-system"
          name="gz::sim::systems::MulticopterMotorModel">
          <robotNamespace>{namespace}</robotNamespace>
          <jointName>{joint}_joint</jointName>
          <linkName>{joint}</linkName>
          <turningDirection>{turning}</turningDirection>
          <maxRotVelocity>{spec['max_rot_velocity']}</maxRotVelocity>
          <motorConstant>{spec['rotor_k']:g}</motorConstant>
          <momentConstant>{_rotor_moment_constant(spec):.10g}</momentConstant>
          <rotorDragCoefficient>8.06428e-05</rotorDragCoefficient>
          <rollingMomentCoefficient>1e-06</rollingMomentCoefficient>
          <rotorVelocitySlowdownSim>20</rotorVelocitySlowdownSim>
          <actuator_number>{actuator}</actuator_number>
          <motorType>velocity</motorType>
        </plugin>''')


def _aero_drag_plugin(spec: dict) -> str:
    """Per-airframe quadratic wind-drag block for the native gz_aero_drag
    plugin. Attached inside the drone's <include> so it loads with the model.
    Only emitted for wind runs; a wind-off world keeps no drag plugin and stays
    byte-identical to the non-aero baseline."""
    return textwrap.dedent(f'''\
          <plugin filename="gz-aero-drag-system"
                  name="gz::sim::systems::AeroDrag">
            <link_name>body</link_name>
            <area>{spec['drag_area_m2']:g}</area>
            <drag_coefficient>{spec['drag_cd']:g}</drag_coefficient>
            <air_density>1.2041</air_density>
          </plugin>''')


def _drone_include(name: str, x: float, y: float, z: float, color: str,
                   drone_mass_kg: float = 0.5, wind: str = 'off') -> str:
    spec = AIRFRAME_SPECS[drone_mass_kg]
    control_extra = _velocity_control_extra(drone_mass_kg, wind)
    aero = _aero_drag_plugin(spec) if wind != 'off' else ''
    motors = '\n'.join(
        _motor_block(name, j, turn, i, spec)
        for i, (j, turn) in enumerate((
            ('propeller_rr', 'ccw'),
            ('propeller_rl', 'cw'),
            ('propeller_fr', 'cw'),
            ('propeller_fl', 'ccw'),
        )))
    rotors = '\n'.join(
        textwrap.dedent(f'''\
          <rotor>
            <jointName>{joint}_joint</jointName>
            <forceConstant>{spec['rotor_k']:g}</forceConstant>
<momentConstant>{_rotor_moment_constant(spec):.10g}</momentConstant>
            <direction>{direction}</direction>
          </rotor>''')
        for (joint, direction) in (
            ('propeller_rr', '1'),
            ('propeller_rl', '-1'),
            ('propeller_fr', '-1'),
            ('propeller_fl', '1'),
        ))
    return textwrap.dedent(f'''\
        <include>
          <uri>model://parrot_bebop_2</uri>
          <name>{name}</name>
          <pose>{x} {y} {z} 0 0 0</pose>

          <!-- No-fly zone cylinder visual. Visual only (no collision) so it acts
               as a visual shield and does not interfere with the trajectory. -->
          <link name="cylinder_link">
            <pose>0 0 0 0 0 0</pose>
            <visual name="visual">
              <geometry>
                <cylinder>
                  <radius>0.75</radius>
                  <length>1.0</length>
                </cylinder>
              </geometry>
              <material>
                <ambient>{color} 1</ambient>
                <diffuse>{color} 1</diffuse>
              </material>
              <transparency>0.5</transparency>
            </visual>
          </link>

          {motors}

          {aero}

          <plugin
            filename="gz-sim-multicopter-control-system"
            name="gz::sim::systems::MulticopterVelocityControl">
            <robotNamespace>{name}</robotNamespace>
            <commandSubTopic>cmd_vel</commandSubTopic>
            <enableSubTopic>enable</enableSubTopic>
            <comLinkName>body</comLinkName>
            <positionGain>5 5.5 3</positionGain>
            {control_extra}
            <rotorConfiguration>
            {rotors}
            </rotorConfiguration>
          </plugin>
        </include>''')


def _target_model(name: str, x: float, y: float, z: float, color: str) -> str:
    return textwrap.dedent(f'''\
        <model name="{name}">
          <pose>{x} {y} {z} 0 0 0</pose>
          <static>true</static>
          <link name="link">
            <visual name="visual">
              <geometry>
                <sphere>
                  <radius>0.2</radius>
                </sphere>
              </geometry>
              <material>
                <ambient>{color} 1</ambient>
                <diffuse>{color} 1</diffuse>
              </material>
            </visual>
          </link>
        </model>''')


def _markers_plugin() -> str:
    """The gz-sim markers system plugin, or empty when the library is absent."""
    if markers_plugin_available():
        return textwrap.dedent('''\
            <plugin filename="gz-sim-markers-system"
                    name="gz::sim::systems::Markers">
            </plugin>''')
    return ''


def _wind_sdf_element(wind: str) -> str:
    """World-level ground-truth wind that gz-sim's SdfEntityCreator turns
    into an ECM wind entity. Without this element the WindEffects system
    finds no wind entity and silently does nothing. The velocity seed comes
    from the option's encoded speed (e.g. 'gust_3' -> 3 m/s along +x)."""
    speed = WIND_SPEED_MS[wind]
    return textwrap.dedent(f'''\
        <wind>
          <linear_velocity>{speed:g} 0 0</linear_velocity>
        </wind>''')


def _wind_plugin(wind: str) -> str:
    """The WindEffects system plugin, or empty when wind == 'off'.

    The wind speed/direction seed comes from the world ``<wind>`` element
    (see ``_wind_sdf_element``); the plugin only low-pass filters it and, for
    'gust_*' settings, adds a sinusoidal gust on the magnitude and direction.

    The plugin's own force factor is zeroed so that it only *generates the wind
    field* (the filtered velocity is read by the native ``gz_aero_drag`` plugin
    which applies the correct per-link quadratic drag instead).
    """
    # Validate argument and build only what that setting needs.
    if wind not in WIND_OPTIONS:
        raise ValueError(f'wind must be one of {WIND_OPTIONS}, got {wind!r}')
    if wind == 'off':
        return ''

    if wind.startswith('gust'):
        magnitude_extra = textwrap.dedent('''\
              <sin>
                <amplitude_percent>0.15</amplitude_percent>
                <period>6.0</period>
              </sin>
              <noise type="gaussian">
                <mean>0</mean>
                <stddev>0.3</stddev>
              </noise>''')
        direction_extra = textwrap.dedent('''\
              <sin>
                <amplitude>5.0</amplitude>
                <period>10.0</period>
              </sin>
              <noise type="gaussian">
                <mean>0</mean>
                <stddev>0.15</stddev>
              </noise>''')
    else:
        magnitude_extra = ''
        direction_extra = ''

    return textwrap.dedent(f'''\
        <plugin filename="gz-sim-wind-effects-system"
                name="gz::sim::systems::WindEffects">
          <!-- Wind field only: force factor 0, so the plugin publishes the
               filtered wind velocity but never exerts its (mass-scaled, linear
               viscous) force. Real drag is applied by gz_aero_drag. -->
          <force_approximation_scaling_factor>0.0</force_approximation_scaling_factor>
          <horizontal>
            <magnitude>
              <time_for_rise>1.0</time_for_rise>
              {magnitude_extra}
            </magnitude>
            <direction>
              <time_for_rise>1.0</time_for_rise>
              {direction_extra}
            </direction>
          </horizontal>
        </plugin>''')


def render_world(num_drones: int = 2, drone_mass_kg: float = 0.5,
                 wind: str = 'off', world_name: str = WORLD_NAME,
                 lane_offset: float = 0.0, altitude_lane: float = 0.3,
                 turbo: bool = False) -> dict:
    """Render the SDF for an N-drone world.

    Deterministic for a given ``num_drones`` (mirrored symmetric placement;
    ``lane_offset`` breaks the coincident central choke point, see
    ``target_positions``; ``altitude_lane`` spreads the crossing vertically
    for N >= 5). ``turbo`` lifts the real-time physics lock so headless
    tuning runs complete as fast as the wall-clock control nodes allow.
    Returns a dict with 'world', 'drone_names', 'starts',
    'targets', 'physics' and 'warnings'.
    """
    if num_drones < 2:
        raise ValueError('num_drones must be >= 2')
    if drone_mass_kg not in DRONE_MASS_OPTIONS:
        raise ValueError(f'drone_mass_kg must be one of {DRONE_MASS_OPTIONS}')
    if wind not in WIND_OPTIONS:
        raise ValueError(f'wind must be one of {WIND_OPTIONS}, got {wind!r}')

    warnings = []
    if wind != 'off':
        gusty = ' (gusty)' if wind.startswith('gust') else ''
        warnings.append(
            f'Wind {WIND_SPEED_MS[wind]:g} m/s along +{WIND_AXIS}{gusty}: '
            'the WindEffects field feeds the native gz_aero_drag quadratic '
            'per-mass drag (gz-sim-wind-effects-system + gz_aero_drag). '
            'Drones spawn at cruise height so takeoff is not contested.')
        if WIND_SPEED_MS[wind] >= 5.0:
            warnings.append(
                'Flight envelope note for 5 m/s: all airframes hold (each '
                'settles at a steady downwind offset - 0.5 kg ~0.6 m, 5 kg '
                '~2-4 m, 20 kg ~5-6 m); gust modes add excursions. 10 m/s is '
                'beyond the command envelope (drones are blown away).')
    if not markers_plugin_available():
        warnings.append(
            'gz-sim markers system library not found: in-sim TEXT name labels '
            'are disabled (drone_labels keeps running; the 2-D GUI trajectory '
            'panel still labels every drone).')
    if drone_mass_kg != 0.5:
        spec = AIRFRAME_SPECS[drone_mass_kg]
        wheelbase_mm = round(2.0 * spec['arm_m'] * 1000.0)
        rpm = int(round(hover_rpm(spec)))
        warnings.append(
            f'Drone size {drone_mass_kg:g} kg: real {spec["name"]}. Flight '
            f'mass {spec["mass_kg"]:g} kg, rotor arm {spec["arm_m"]:.3f} m '
            f'(wheelbase {wheelbase_mm} mm), rotor thrust constant '
            f'{spec["rotor_k"]:g}, hover ~{rpm} RPM, motor cap '
            f'{spec["max_rot_velocity"]:g} rad/s. Heavier drones fly '
            'sedately like a commercial UAV: gentler cruise speed, capped '
            'acceleration and a deceleration ramp into the target.')

    phys = physics_config(num_drones, turbo)
    if phys['real_time_factor'] > 1.0:
        warnings.append(
            f'Turbo mode: physics is not locked to real time (RTF up to '
            f'{phys["real_time_factor"]:g}x). The wall-clock control nodes '
            'still tick in real time, so results come from the saved stats/'
            'tlogs, not the live GUI elapsed clock.')
    elif phys['degraded']:
        warnings.append(
            f'Physics relaxed for {num_drones} drones: '
            f'{phys["real_time_update_rate"]} Hz, step {phys["max_step_size"]}, '
            f'solver iters {phys["solver_iters"]} (keeps the session close to '
            'real-time and frees CPU for the avoidance nodes).')

    xs = spawn_x(num_drones)
    spawn_z = START_Z_WIND if wind != 'off' else START_Z
    starts = [(x, START_Y, spawn_z) for x in xs]
    targets = target_positions(xs, lane_offset, altitude_lane)
    names = [f'drone_{i}' for i in range(num_drones)]

    targets_xml = '\n\n'.join(
        _target_model(f'target_{i}', tx, ty, tz, COLORS[i % len(COLORS)])
        for i, (tx, ty, tz) in enumerate(targets))
    drones_xml = '\n\n'.join(
        _drone_include(names[i], *starts[i], COLORS[i % len(COLORS)],
                       drone_mass_kg, wind)
        for i in range(num_drones))

    body = textwrap.dedent(f'''\
        <sdf version="1.8">
        <world name="{world_name}">
        <physics name="default_physics" type="ode">
          <max_step_size>{phys['max_step_size']}</max_step_size>
          <real_time_factor>{phys['real_time_factor']:g}</real_time_factor>
          <real_time_update_rate>{phys['real_time_update_rate']}</real_time_update_rate>
          <ode>
            <solver>
              <type>world</type>
              <iters>{phys['solver_iters']}</iters>
              <sor>1.0</sor>
            </solver>
            <constraints>
              <cfm>0.00001</cfm>
              <erp>0.2</erp>
            </constraints>
          </ode>
        </physics>
        <gravity>0 0 -9.8</gravity>
        <magnetic_field>6e-06 2.3e-05 -4.2e-05</magnetic_field>
        <atmosphere type="adiabatic"/>
        {_wind_sdf_element(wind) if wind != 'off' else ''}

        <!-- Gazebo Harmonic (gz-sim 8) system plugins. -->
        <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics">
        </plugin>
        <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands">
        </plugin>
        <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster">
        </plugin>
        <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors">
        </plugin>
        {_markers_plugin()}
        {_wind_plugin(wind)}
        <scene>
          <ambient>0.4 0.4 0.4 1.0</ambient>
          <background>0.7 0.7 0.8 1.0</background>
          <grid>true</grid>
          <shadows>true</shadows>
        </scene>

        <light name="sun" type="directional">
          <cast_shadows>true</cast_shadows>
          <pose>0 0 10 0 0 0</pose>
          <diffuse>1 1 1 1</diffuse>
          <specular>0.2 0.2 0.2 1</specular>
          <direction>-0.5 0.1 -0.9</direction>
        </light>

        <model name="ground_plane">
          <static>true</static>
          <link name="link">
            <collision name="collision">
              <geometry>
                <plane>
                  <normal>0 0 1</normal>
                  <size>100 100</size>
                </plane>
              </geometry>
            </collision>
            <visual name="visual">
              <geometry>
                <plane>
                  <normal>0 0 1</normal>
                  <size>100 100</size>
                </plane>
              </geometry>
              <material>
                <ambient>0.2 0.2 0.2 1</ambient>
                <diffuse>0.2 0.2 0.2 1</diffuse>
              </material>
            </visual>
          </link>
        </model>

        <!-- Destination spheres (one per drone, visual-only so the drones can
             pass straight through their targets). -->
        {targets_xml}

        <!-- Drones: one param-bebop include per drone. -->
        {drones_xml}
        </world>
        </sdf>''')

    # Uniform base indent (declaration must stay at column 0).
    world = '<?xml version="1.0" ?>\n' + textwrap.indent(body.strip(), '    ')

    return {
        'world': world,
        'drone_names': names,
        'starts': starts,
        'targets': targets,
        'physics': phys,
        'warnings': warnings,
    }


def render_bridge(drone_names: list) -> str:
    """Render the ros_gz_bridge YAML for the given drone names."""
    lines = [
        '# Auto-generated by world_generator.py (N drones).',
        '#',
        '# Bridges each drone\'s fused velocity command (ROS Twist) to the',
        '# gz.msgs.Twist topic of its MulticopterVelocityControl system, plus',
        '# the motor enable signal used to arm/disarm the rotors.',
    ]
    for name in drone_names:
        lines += [
            f'- ros_topic_name: "/{name}/drive_cmd"',
            f'  gz_topic_name: "/{name}/cmd_vel"',
            '  ros_type_name: "geometry_msgs/msg/Twist"',
            '  gz_type_name: "gz.msgs.Twist"',
            '  direction: "ROS_TO_GZ"',
            '',
            f'- ros_topic_name: "/{name}/enable"',
            f'  gz_topic_name: "/{name}/enable"',
            '  ros_type_name: "std_msgs/msg/Bool"',
            '  gz_type_name: "gz.msgs.Boolean"',
            '  direction: "ROS_TO_GZ"',
            '',
        ]
    return '\n'.join(lines)


def _strip_sensors(sdf: str) -> str:
    """Remove the multiranger GPU-lidar and fisheye camera sensor blocks.

    The CAS stack is GPS-only; at scale (10 drones) the bundled camera +
    GPU-lidar per drone load the sensors/rendering system for topics nobody
    subscribes to. Keeps every other element (mesh, collisions, props,
    plugin) byte-identical so the world still reflects the real airframe.
    """
    out = []
    i = 0
    while True:
        start = sdf.find('<sensor', i)
        if start == -1:
            out.append(sdf[i:])
            break
        out.append(sdf[i:start])
        depth = 0
        j = start
        while True:
            j = sdf.find('</sensor>', j)
            if j == -1:
                out.append(sdf[start:])
                i = len(sdf)
                break
            close = j + len('</sensor>')
            # verify this closes the sensor we opened (no nested sensors)
            if sdf.count('<sensor', start, close) - 1 == depth:
                i = close
                break
            depth += 1
            j = close
        if i >= len(sdf):
            break
    return ''.join(out)


def generate_files(output_dir, num_drones=2, drone_mass_kg=0.5, wind='off',
                   world_name=WORLD_NAME, lane_offset=0.0,
                   altitude_lane=0.3, drone_model='full',
                   turbo=False) -> dict:
    """Render and write the world + bridge files; return their paths.

    For ``drone_mass_kg`` != 0.5 a physics rewrite of the Parrot model is
    also written under ``output_dir/models/<mass>kg/parrot_bebop_2/model.sdf``
    so ``model://parrot_bebop_2`` resolves to the real-airframe body
    mass/inertia and propeller geometry (see AIRFRAME_SPECS). For
    ``drone_model='lite'`` an overlay without the (GPS-unused) camera /
    GPU-lidar sensors is written under ``output_dir/models/lite/...``.
    Returns 'scaled_mass' (the kg value when a copy was written, else None)
    and 'model_dir' (dir to prepend to GZ_SIM_RESOURCE_PATH, else None).
    """
    os.makedirs(output_dir, exist_ok=True)
    data = render_world(num_drones, drone_mass_kg, wind, world_name,
                        lane_offset, altitude_lane, turbo)
    world_path = os.path.join(output_dir, f'world_{num_drones}.world')
    bridge_path = os.path.join(output_dir, f'bridge_{num_drones}.yaml')
    with open(world_path, 'w') as f:
        f.write(data['world'])
    with open(bridge_path, 'w') as f:
        f.write(render_bridge(data['drone_names']))
    data['world_path'] = world_path
    data['bridge_path'] = bridge_path

    subdir = f'{drone_mass_kg:g}kg' if drone_mass_kg != 0.5 else 'lite'
    if drone_mass_kg == 0.5 and drone_model != 'lite':
        data['scaled_mass'] = None
        data['model_dir'] = None
        data['lite_model'] = False
        return data

    model_dir = os.path.join(output_dir, 'models', subdir)
    model_path = os.path.join(model_dir, 'parrot_bebop_2', 'model.sdf')
    base_model = os.path.dirname(_model_sdf_path())
    shutil.copytree(base_model, os.path.dirname(model_path),
                    dirs_exist_ok=True)
    with open(os.path.join(base_model, 'model.sdf')) as f:
        sdf_text = f.read()
    if drone_mass_kg != 0.5:
        sdf_text = _rewrite_model_sdf(sdf_text,
                                      AIRFRAME_SPECS[drone_mass_kg])
    if drone_model == 'lite':
        sdf_text = _strip_sensors(sdf_text)
    with open(model_path, 'w') as f:
        f.write(sdf_text)
    data['scaled_mass'] = drone_mass_kg if drone_mass_kg != 0.5 else None
    data['model_dir'] = model_dir
    data['lite_model'] = drone_model == 'lite'
    return data


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Generate an N-drone Gazebo Harmonic world + bridge config.')
    parser.add_argument('--num-drones', type=int, default=2,
                        help='Number of drones (>= 2).')
    parser.add_argument('--mass', type=float, default=0.5,
                        choices=DRONE_MASS_OPTIONS,
                        help='Drone mass in kg (0.5 = real Parrot Bebop 2).')
    parser.add_argument('--wind', choices=WIND_OPTIONS, default='off',
                        help='Wind via the WindEffects field + native '
                             'gz_aero_drag per-mass drag. Each non-off value '
                             'encodes mode + m/s speed, e.g. strong_5 '
                             '(constant 5 m/s) or gust_3 (3 m/s gusty).')
    parser.add_argument('--lane-offset', type=float, default=0.0,
                        help='Break the symmetric single-point crossing '
                             '(alternate target x shifted +/- this many m).')
    parser.add_argument('--drone-model', choices=['full', 'lite'], default='full',
                        help="'full' = bundled Bebop model (camera + GPU-lidar); "
                             "'lite' = overlay without the GPS-unused sensors "
                             '(cuts sensor load at scale).')
    parser.add_argument('--altitude-lane', type=float, default=0.3,
                        help='Target-z altitude ladder spacing in m for N >= 5 '
                             '(0 disables; spreads the single-point crossing '
                             'vertically so dense flocks can pass).')
    parser.add_argument('--output-dir', default='/tmp/unikanie_kolizji_gen',
                        help='Writable directory for the generated files.')
    parser.add_argument('--turbo', action='store_true',
                        help='Lift the real-time physics lock (RTF up to 1000x) '
                             'for headless/fast tuning runs; results are read '
                             'from saved stats/tlogs, not the live GUI.')
    args = parser.parse_args()

    data = generate_files(args.output_dir, args.num_drones, args.mass, args.wind,
                          lane_offset=args.lane_offset,
                          altitude_lane=args.altitude_lane,
                          drone_model=args.drone_model, turbo=args.turbo)
    for w in data['warnings']:
        print(f'[world_generator] warning: {w}', file=__import__('sys').stderr)
    print(f'[world_generator] wrote {data["world_path"]}')
    print(f'[world_generator] wrote {data["bridge_path"]}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())