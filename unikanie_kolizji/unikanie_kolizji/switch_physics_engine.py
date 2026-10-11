#!/usr/bin/env python3

import argparse
import os
import shutil
import subprocess
import sys


# Physics engines available in Gazebo Harmonic (gz-sim 8) via gz-physics7.
#
# The user-facing engine selector is mapped to the actual gz-physics plugin
# library name. The gz-sim '--physics-engine' flag expects the plugin LIBRARY
# name (e.g. 'gz-physics7-ode-plugin'), not the bare engine name, otherwise it
# fails with "Failed to find plugin".
#
# NOTE: ODE is required for the MulticopterVelocityControl system used by the
# Bebop 2 drone models. TPE and other engines may not work with multicopter
# motor mixing.
PHYSICS_ENGINE_OPTIONS = [
    'ode',
    'tpe',
    'bullet',
    'dart',
]

ENGINE_PLUGIN_MAP = {
    'ode': 'gz-physics7-ode-plugin',
    'tpe': 'gz-physics7-tpe-plugin',
    'bullet': 'gz-physics7-bullet-plugin',
    'dart': 'gz-physics7-dartsim-plugin',
}

# Directory that contains the gz-physics engine plugins. The Physics system
# scans GZ_SIM_PHYSICS_ENGINE_PATH for the engine library.
ENGINE_PLUGIN_PATHS = [
    '/opt/ros/jazzy/opt/gz_physics_vendor/lib/gz-physics-7/engine-plugins',
    '/usr/lib/x86_64-linux-gnu/gz-physics-7/engine-plugins',
]


def resolve_engine_plugin_path() -> str | None:
    for p in ENGINE_PLUGIN_PATHS:
        if os.path.isdir(p):
            return p
    return None


def build_gz_command(engine_name: str, world_file: str) -> list[str]:
    plugin_name = ENGINE_PLUGIN_MAP.get(engine_name, engine_name)
    cmd = [
        'gz', 'sim',
        '-v', '4',
        '-r',
        '--headless-rendering',
        '--physics-engine', plugin_name,
        world_file,
    ]
    return cmd


def gz_env() -> dict:
    env = dict(os.environ)
    path = resolve_engine_plugin_path()
    if path:
        env['GZ_SIM_PHYSICS_ENGINE_PATH'] = path
    return env


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Launch a Gazebo Harmonic world with a chosen physics engine.')
    parser.add_argument(
        '--engine',
        choices=PHYSICS_ENGINE_OPTIONS,
        default='ode',
        help='Gazebo Harmonic physics engine to use (default: ode).')
    parser.add_argument(
        '--world',
        default=os.path.join(
            os.path.dirname(__file__), '..', 'worlds',
            'empty_world_harmonic.world'),
        help='Path to the Gazebo world file.')
    args = parser.parse_args()

    world_path = os.path.abspath(args.world)
    if not os.path.exists(world_path):
        print(f'World file not found: {world_path}', file=sys.stderr)
        return 1
    if shutil.which('gz') is None:
        print('gz executable not found on PATH.', file=sys.stderr)
        return 1

    cmd = build_gz_command(args.engine, world_path)
    print('Launching Gazebo Harmonic with command:')
    print(' '.join(cmd))
    return subprocess.call(cmd, env=gz_env())


if __name__ == '__main__':
    raise SystemExit(main())
