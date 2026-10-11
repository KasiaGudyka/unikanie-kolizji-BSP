#!/usr/bin/env python3

"""Option compatibility layer for the drone simulation configuration GUI.

Pure-Python rules that drive the GUI selectors and encode the SAFETY
precautions: not every combination of physics engine, drone size, wind,
drone count and collision avoidance method is valid.  The GUI calls
:func:`allowed_engines` and :func:`validate` to disable incompatible options
in real time and to explain why.

The five option groups drive the simulation configuration:
  1. physics engine  (Gazebo Harmonic UAV simulation)
  2. drone size      (0.5 kg, 5 kg, 20 kg)
  3. strong wind     (off / strong)
  4. drone count     (2, 5, 10)
  5. CAS method      (collision avoidance: GPS / 3D-SWAP / DPSO)
"""

import dataclasses
from dataclasses import dataclass
from typing import List, Tuple

from .switch_physics_engine import PHYSICS_ENGINE_OPTIONS


# --- Option domain ---------------------------------------------------------

# Drone masses in kg. The 0.5 kg entry is the real Parrot Bebop 2 model
# (pristine model.sdf). The 5 kg / 20 kg variants reuse the same geometry
# with the physics of a real airframe of that weight class, written at launch
# time as a model rewrite (see world_generator.AIRFRAME_SPECS).
DRONE_MASS_OPTIONS: Tuple[float, ...] = (0.5, 5.0, 20.0)

# Physics engines usable for a Gazebo Harmonic (gz-sim 8) UAV simulation.
# Kept in sync with switch_physics_engine.PHYSICS_ENGINE_OPTIONS.
ENGINE_OPTIONS: Tuple[str, ...] = tuple(PHYSICS_ENGINE_OPTIONS)

# Number of drones the world generator (world_generator.py) can spawn per
# session; the launch file regenerates the world accordingly.
DRONE_COUNT_OPTIONS: Tuple[int, ...] = (2, 5, 10)

# Wind settings (implemented via gz-sim WindEffects + the native gz_aero_drag
# system plugin). Every non-'off' value encodes the mode and the wind speed
# in m/s:
#   'strong_N'  = uniform N m/s cross-wind (constant), along +x.
#   'gust_N'    = same N m/s seed but the plugin adds a sinusoidal gust /
#                 directional wobble to the magnitude.
# WindEffects only generates the filtered wind field (its own force is zeroed);
# gz_aero_drag then applies the per-airframe quadratic drag law
# F = 0.5 * rho * Cd * A * |v_rel| * v_rel, which is area-based and independent
# of body mass - so 1-5 m/s pushes the 0.5 kg toy noticeably but barely moves a
# 20 kg field quad, unlike the old mass-scaled linear blow-away model.
# Flight envelope: 1 m/s is imperceptible at every mass; 3 m/s stays within
# every profile; 5 m/s is held by every airframe - each settles at a steady
# downwind offset (0.5 kg ~0.6 m, 5 kg ~2-4 m, 20 kg ~5-6 m) and rivets its
# target row instead of reaching the exact target sphere; gust modes add
# excursions and 10 m/s exceeds the command envelope (drones are blown away).
# Wind worlds spawn the drones at cruise height, arm them on spawn and gate
# readiness on stable altitude (never on the start x/y, which would blow open):
# the gz velocity-only controller has no position hold against a constant
# wind, so the cart ahead of the flight controller pulverizes the wind fight,
# and body-frame commands track through any heading (MulticopterVelocityControl
# feeds its velocity error with R * cmd).
WIND_OPTIONS: Tuple[str, ...] = (
    'off',
    'strong_1', 'strong_3', 'strong_5', 'strong_10',
    'gust_1', 'gust_3', 'gust_5', 'gust_10',
)
WIND_SPEED_MS: dict = {w: 0.0 if w == 'off' else float(w.split('_', 1)[1])
                       for w in WIND_OPTIONS}

# Collision Avoidance System (CAS) method the user wants to test.
# Each key maps to a named method in drone_collision_avoidance.METHODS.
# '3d_swap' = 3D-SWAP velocity obstacle   (method_2, implemented; default
#             for crossing-path scenarios)
# 'gps'     = GPS / no-fly-zone repulsion (method_1, implemented)
# 'dpso'    = distributed PSO trajectory planner (method_3, implemented)
# 'convex'  = convex-hull decongestion    (method_4, implemented)
# 'gmm'     = Gaussian-mixture avoidance  (method_5, implemented)
# 'combined'= DPSO + GMM + convex hull    (method_6, implemented)
CAS_OPTIONS: Tuple[str, ...] = ('3d_swap', 'gps', 'dpso', 'convex', 'gmm',
                                'combined')

# User-facing CAS names → launch-file 'collision_avoidance_method' values.
# The launch file (gazebo_two_drones.launch.py) and drone nodes still speak
# the legacy method_1/2/3 vocabulary, so the GUI must translate.
CAS_TO_METHOD: dict = {
    'gps': 'method_1',
    '3d_swap': 'method_2',
    'dpso': 'method_3',
    'convex': 'method_4',
    'gmm': 'method_5',
    'combined': 'method_6',
}


def cas_launch_value(cas: str) -> str:
    """Translate a CAS_OPTIONS name to the launch's collision_avoidance_method."""
    return CAS_TO_METHOD.get(cas, cas)

# --- Compatibility matrix (safety precautions) -----------------------------
#
# MulticopterVelocityControl / MulticopterMotorModel are only verified with the
# ODE physics engine, so the 0.5 kg Parrot Bebop 2 may only use ODE. Heavier
# drones are assumed to tolerate the other engines until proven otherwise.

ALLOWED_ENGINES_FOR_MASS: dict = {
    0.5: ['ode'],
    5.0: list(ENGINE_OPTIONS),
    20.0: list(ENGINE_OPTIONS),
}


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of a compatibility check.

    - `valid`: True when there is no blocking error (warnings allowed).
    - `errors`: blocking problems; the selected combination must not run.
    - `warnings`: non-blocking safety hints the operator should read.
    """

    valid: bool
    errors: List[str] = dataclasses.field(default_factory=list)
    warnings: List[str] = dataclasses.field(default_factory=list)


def allowed_engines(drone_mass: float) -> List[str]:
    """Return the physics engines compatible with ``drone_mass`` (kg)."""
    allowed = ALLOWED_ENGINES_FOR_MASS.get(drone_mass)
    if allowed is None:
        return []
    return list(allowed)


def _validate_engine(drone_mass: float, engine: str) -> List[str]:
    allowed = allowed_engines(drone_mass)
    if engine not in allowed:
        return [f'{_mass_label(drone_mass)} supports only '
                f'{", ".join(allowed)}; "{engine}" is not compatible.']
    return []


def _validate_wind(drone_mass: float, wind: str) -> List[str]:
    if wind == 'off':
        return []
    speed = WIND_SPEED_MS[wind]
    if drone_mass == 0.5 and speed >= 5.0:
        gust = ' (gusts included)' if wind.startswith('gust') else ''
        return [f'Wind {speed:g} m/s'
                f'{" (gusty)" if gust else ""} will strongly perturb the '
                '0.5 kg drone; the inertia-scaled controller keeps it '
                'controllable and a strong wind is the intended challenge.']
    return []


def _validate_count(drone_mass: float, count: int) -> List[str]:
    if drone_mass == 0.5 and count >= 5:
        return ['More than 5 lightweight 0.5 kg drones in one world may '
                'tax the real-time solver; expect reduced simulation '
                'performance.']
    return []


def _mass_label(drone_mass: float) -> str:
    return f'{drone_mass:g} kg drone'


def validate(drone_mass: float, engine: str, wind: str = 'off',
             drone_count: int = 2, cas: str = 'gps') -> ValidationResult:
    """Check a full option selection against the compatibility matrix."""
    errors = []
    warnings = []

    if drone_mass not in DRONE_MASS_OPTIONS:
        errors.append(f'Unsupported drone mass "{drone_mass}".')
    if engine not in ENGINE_OPTIONS:
        errors.append(f'Unsupported engine "{engine}".')
    if wind not in WIND_OPTIONS:
        errors.append(f'Unsupported wind setting "{wind}".')
    if drone_count not in DRONE_COUNT_OPTIONS:
        errors.append(f'Unsupported drone count "{drone_count}".')
    if cas not in CAS_OPTIONS:
        errors.append(f'Unsupported CAS method "{cas}".')

    if not errors:
        errors += _validate_engine(drone_mass, engine)
        warnings += _validate_wind(drone_mass, wind)
        warnings += _validate_count(drone_mass, drone_count)

    return ValidationResult(valid=not errors, errors=errors, warnings=warnings)


def resolve(drone_mass: float, engine: str, wind: str = 'off',
            drone_count: int = 2, cas: str = 'gps') -> dict:
    """Return a validated selection, forcing invalid engines to a safe default.

    Used by the GUI to self-correct: e.g. changing the drone size to 0.5 kg
    while TPE is selected resolves the engine back to ODE, the only engine the
    lightest drone may use.
    """
    allowed = allowed_engines(drone_mass)
    if allowed and engine not in allowed:
        engine = allowed[0]
    result = validate(drone_mass, engine, wind, drone_count, cas)
    return {
        'drone_mass': drone_mass,
        'engine': engine,
        'wind': wind,
        'drone_count': drone_count,
        'cas': cas,
        'validation': result,
    }