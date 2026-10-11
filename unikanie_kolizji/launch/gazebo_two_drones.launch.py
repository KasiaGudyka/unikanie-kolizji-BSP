#!/usr/bin/env python3

import os
import random
import shutil
import tempfile

from ament_index_python.packages import (get_package_prefix,
                                         get_package_share_directory)
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction, \
    RegisterEventHandler, Shutdown
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from unikanie_kolizji import world_generator
from unikanie_kolizji import combined_avoid
from unikanie_kolizji import convex_hull_avoid
from unikanie_kolizji import dpso_swarm
from unikanie_kolizji import gmm_avoid
from unikanie_kolizji.switch_physics_engine import ENGINE_PLUGIN_MAP


def _launch_setup(context):
    render_engine = LaunchConfiguration('render_engine').perform(context)
    physics_engine = LaunchConfiguration('physics_engine').perform(context)
    drone_count = int(LaunchConfiguration('drone_count').perform(context))
    drone_size_kg = float(LaunchConfiguration('drone_size_kg').perform(context))
    wind = LaunchConfiguration('wind').perform(context)
    cas = LaunchConfiguration('collision_avoidance_method').perform(context)
    collision_radius = float(
        LaunchConfiguration('collision_radius').perform(context))
    lane_offset = float(LaunchConfiguration('lane_offset').perform(context))
    altitude_lane = float(LaunchConfiguration('altitude_lane').perform(context))
    drone_model = LaunchConfiguration('drone_model').perform(context)
    recording_enabled = (LaunchConfiguration('recording_enabled')
                         .perform(context).strip().lower()
                         in ('true', '1', 'yes'))
    perf_probe = (LaunchConfiguration('perf_probe')
                  .perform(context).strip().lower()
                  in ('true', '1', 'yes'))
    perf_dir = LaunchConfiguration('perf_dir').perform(context).strip()


    def _num(name, default, cast=float):
        """Launch-config helper: '' / 'none' falls back to the default."""
        raw = LaunchConfiguration(name).perform(context).strip()
        try:
            return cast(raw) if raw and raw != 'none' else default
        except ValueError:
            return default

    dp = dpso_swarm.DPSOParams
    hp = convex_hull_avoid.HullParams
    gp = gmm_avoid.GMMParams
    cp = combined_avoid.CombinedParams
    cas_tunables = {
        'dpso_pop': _num('dpso_pop', dp.pop, int),
        'dpso_max_iter': _num('dpso_max_iter', dp.max_iter, int),
        'dpso_horizon': _num('dpso_horizon', dp.horizon, int),
        'dpso_plan_dt': _num('dpso_plan_dt', dp.plan_dt),
        'dpso_replan_ticks': _num('dpso_replan_ticks', dp.replan_ticks, int),
        'dpso_w': _num('dpso_w', dp.w),
        'dpso_c1': _num('dpso_c1', dp.c1),
        'dpso_c2': _num('dpso_c2', dp.c2),
        'dpso_w_goal': _num('dpso_w_goal', dp.w_goal),
        'dpso_w_ob': _num('dpso_w_ob', dp.w_ob),
        'dpso_w_effort': _num('dpso_w_effort', dp.w_effort),
        'dpso_w_smooth': _num('dpso_w_smooth', dp.w_smooth),
        'dpso_keepout': _num('dpso_keepout', dp.keepout),
        'dpso_ob_stop': _num('dpso_ob_stop', dp.ob_stop),
        'dpso_speed_cap': _num('dpso_speed_cap', dp.speed_cap),
        'hull_activation': _num('hull_activation', hp.activation),
        'hull_gain': _num('hull_gain', hp.gain),
        'hull_base_speed': _num('hull_base_speed', hp.base_speed),
        'hull_speed_cap': _num('hull_speed_cap', hp.speed_cap),
        'gmm_sigma': _num('gmm_sigma', gp.sigma),
        'gmm_gain': _num('gmm_gain', gp.gain),
        'gmm_activation': _num('gmm_activation', gp.activation),
        'gmm_base_speed': _num('gmm_base_speed', gp.base_speed),
        'gmm_speed_cap': _num('gmm_speed_cap', gp.speed_cap),
        'comb_w_gmm': _num('comb_w_gmm', cp.comb_w_gmm),
        'stgo_adoption': _num('stgo_adoption', cp.stgo_adoption),
    }
    seed = LaunchConfiguration('seed').perform(context)
    goal_timeout_raw = LaunchConfiguration('goal_timeout').perform(context).strip()
    goal_timeout = (float(goal_timeout_raw) if goal_timeout_raw else
                    (600.0 if wind != 'off' else 300.0))
    if seed in ('', 'auto'):
        seed = str(random.SystemRandom().randint(100000, 999999))

    headless = render_engine == 'headless'
    show_labels = (LaunchConfiguration('show_labels').perform(context).strip()
                   .lower() in ('true', '1', 'yes'))
    # Headless = autonomous tuning run: lift the real-time physics lock so the
    # whole flight completes as fast as the wall-clock control nodes allow
    # (results judged from saved stats/tlogs). Live GUI runs stay 1:1.
    turbo = headless

    pkg_share = get_package_share_directory('unikanie_kolizji')
    gz_bin = shutil.which('gz') or '/opt/ros/jazzy/opt/gz_tools_vendor/bin/gz'


    def _aero_plugin_lib_dir():
        """lib dir of the native gz_aero_drag plugin (same workspace)."""
        try:
            return os.path.join(get_package_prefix('gz_aero_drag'),
                                'lib', 'gz_aero_drag')
        except Exception:
            return ''

    aero_lib = _aero_plugin_lib_dir()

    # Shared simulator environment.
    sim_env = {
        **os.environ,
        'GZ_SIM_PHYSICS_ENGINE_PATH':
            '/opt/ros/jazzy/opt/gz_physics_vendor/lib/gz-physics-7/engine-plugins:'
            '/usr/lib/x86_64-linux-gnu/gz-physics-7/engine-plugins',
        'GZ_SIM_RENDER_ENGINE_PATH':
            '/opt/ros/jazzy/opt/gz_rendering_vendor/lib/gz-rendering-8/engine-plugins'
            + (os.pathsep + os.environ['GZ_SIM_RENDER_ENGINE_PATH']
               if os.environ.get('GZ_SIM_RENDER_ENGINE_PATH') else ''),
        'GZ_SIM_RESOURCE_PATH':
            os.path.join(pkg_share, 'models')
            + (os.pathsep + os.environ['GZ_SIM_RESOURCE_PATH']
               if os.environ.get('GZ_SIM_RESOURCE_PATH') else ''),
        'GZ_SIM_SYSTEM_PLUGIN_PATH':
            aero_lib
            + (os.pathsep + os.environ['GZ_SIM_SYSTEM_PLUGIN_PATH']
               if aero_lib and os.environ.get('GZ_SIM_SYSTEM_PLUGIN_PATH')
               else ''),
    }

    # Generate a fresh N-drone world + matching bridge config for this session.
    gen_dir = os.path.join(tempfile.gettempdir(), 'unikanie_kolizji_gen')
    data = world_generator.generate_files(
        gen_dir, num_drones=drone_count, drone_mass_kg=drone_size_kg, wind=wind,
        lane_offset=lane_offset, altitude_lane=altitude_lane,
        drone_model=drone_model, turbo=turbo)
    world_file = data['world_path']
    bridge_config = data['bridge_path']
    for warning in data['warnings']:
        print(f'[launch] world generator warning: {warning}')

    # Heavier drone sizes rewrite the Parrot model with the physics of a real
    # airframe (mass/inertia + real wheelbase rotor geometry) under the
    # generation dir; prepend it so model://parrot_bebop_2 resolves to it.
    if data.get('model_dir'):
        sim_env['GZ_SIM_RESOURCE_PATH'] = (
            data['model_dir'] + os.pathsep + sim_env['GZ_SIM_RESOURCE_PATH'])

    # Pass a per-run COPY of the GUI config, not the shipped file: gz-gui
    # rewrites its --gui-config file (camera pose, layout) on exit, which
    # would otherwise silently corrupt the installed config.
    gui_config = os.path.join(pkg_share, 'config', 'gui.config')
    session_gui_config = os.path.join(gen_dir, 'gui.config')
    shutil.copy2(gui_config, session_gui_config)
    gui_config = session_gui_config

    engine_flags = []
    if physics_engine != 'ode':
        plugin = ENGINE_PLUGIN_MAP.get(physics_engine)
        if plugin:
            engine_flags = ['--physics-engine', plugin]
        else:
            print(f'[launch] unknown physics engine {physics_engine!r}, '
                  'keeping the world default (ode).')

    # -z <seed> runs every session in a transient world (unique per run), so a
    # stale leftover gz-server can no longer leak models into a new launch.
    base_cmd = [gz_bin, 'sim', '-v', '4', '-r', '-z', seed, *engine_flags]

    if headless:
        gz_cmd = base_cmd + ['-s', '--headless-rendering', world_file]
        gz_extra_env = {'QT_QPA_PLATFORM': 'offscreen'}
    else:
        gz_cmd = base_cmd + ['--gui-config', gui_config,
                             '--render-engine', render_engine, world_file]
        gz_extra_env = {}

    gz_action = ExecuteProcess(
        cmd=gz_cmd,
        output='screen',
        env={**sim_env, **gz_extra_env},
    )

    actions = [
        gz_action,
        RegisterEventHandler(
            OnProcessExit(
                target_action=gz_action,
                on_exit=[Shutdown()],
            )
        ),
    ]

    names = data['drone_names']
    starts = data['starts']
    targets = data['targets']
    flat_targets = [v for target in targets for v in target]
    authority = {0.5: 1.0, 5.0: 0.7, 20.0: 0.65}[drone_size_kg]

    for i, name in enumerate(names):
        actions.append(Node(
            package='unikanie_kolizji',
            executable='drone_gps',
            name=f'drone_gps_{i}',
            parameters=[{
                'model_name': name,
                'topic_prefix': name,
                'target_world': list(targets[i]),
                'start_world': list(starts[i]),
                'drone_mass_kg': drone_size_kg,
                'collision_avoidance_method': cas,
                'wind': wind,
                'neighbor_prefixes': [n for n in names if n != name],
                'collision_radius': collision_radius,
            }],
            output='screen',
        ))

    actions.append(Node(
        package='unikanie_kolizji',
        executable='drone_start_gate',
        name='drone_start_gate',
        parameters=[{
            'drone_prefixes': names,
            'goal_timeout': goal_timeout,
            'recording_enabled': recording_enabled,
        }],
        output='screen',
    ))

    actions.append(Node(
        package='unikanie_kolizji',
        executable='drone_collision_avoidance',
        name='drone_collision_avoidance',
        parameters=[{
            'collision_avoidance_method': cas,
            'drone_prefixes': names,
            'targets': flat_targets,
            'mass_authorities': [authority] * len(names),
            'perf_probe': perf_probe,
            'perf_dir': perf_dir,
            **cas_tunables,
        }],
        output='screen',
    ))

    if show_labels:
        actions.append(Node(
            package='unikanie_kolizji',
            executable='drone_labels',
            name='drone_labels',
            parameters=[{
                'drone_names': names,
                'targets': flat_targets,
                'world_name': world_generator.WORLD_NAME,
            }],
            output='screen',
        ))

    # Bridge every drone's fused velocity command to its MulticopterVelocityControl
    # topic, arm/disarm enabled via the Bool enable topics.
    actions.append(Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='drone_velocity_bridge',
        parameters=[{'config_file': bridge_config}],
        output='screen',
    ))

    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'collision_avoidance_method',
            default_value='method_2',
            choices=['method_1', 'method_2', 'method_3',
                     'method_4', 'method_5', 'method_6'],
            description='Collision avoidance method. method_2 (3D-SWAP '
                        'roundabout) is the recommended default for crossing '
                        'paths; method_1 (GPS/no-fly-zone repulsion) is the '
                        'buffer default; method_3 (DPSO) is a receding-horizon '
                        'particle-swarm trajectory planner; method_4 is '
                        'convex-hull decongestion; method_5 is Gaussian-mixture '
                        'density avoidance; method_6 combines DPSO + GMM + '
                        'convex hull.'),

        DeclareLaunchArgument(
            'recording_enabled',
            default_value='true',
            description='Write the .tlog telemetry log and the ros2 bag for '
                        'the run to disk (false skips both recorders).'),

        DeclareLaunchArgument(
            'perf_probe',
            default_value='false',
            description='Instrument drone_collision_avoidance per-tick wall/CPU '
                        'timing and save perf_<method>.csv/.json under perf_dir '
                        '(for the avoidance-method benchmark).'),

        DeclareLaunchArgument(
            'perf_dir',
            default_value='',
            description='Directory for the perf_probe csv/json output '
                        '(empty = save nothing).'),

        DeclareLaunchArgument(
            'dpso_pop',
            default_value=str(dpso_swarm.DPSOParams.pop),
            description='DPSO (method_3/6): particles per drone swarm.'),
        DeclareLaunchArgument(
            'dpso_max_iter',
            default_value=str(dpso_swarm.DPSOParams.max_iter),
            description='DPSO (method_3/6): max iteration times M per replan.'),
        DeclareLaunchArgument(
            'dpso_horizon',
            default_value=str(dpso_swarm.DPSOParams.horizon),
            description='DPSO (method_3/6): planning horizon (velocity steps).'),
        DeclareLaunchArgument(
            'dpso_plan_dt',
            default_value=str(dpso_swarm.DPSOParams.plan_dt),
            description='DPSO (method_3/6): seconds per horizon step.'),
        DeclareLaunchArgument(
            'dpso_replan_ticks',
            default_value=str(dpso_swarm.DPSOParams.replan_ticks),
            description='DPSO (method_3/6): replan every N ticks (20 Hz).'),
        DeclareLaunchArgument(
            'dpso_w',
            default_value=str(dpso_swarm.DPSOParams.w),
            description='DPSO (method_3/6): PSO inertia weight.'),
        DeclareLaunchArgument(
            'dpso_c1',
            default_value=str(dpso_swarm.DPSOParams.c1),
            description='DPSO (method_3/6): cognitive (pbest) coefficient.'),
        DeclareLaunchArgument(
            'dpso_c2',
            default_value=str(dpso_swarm.DPSOParams.c2),
            description='DPSO (method_3/6): social (gbest) coefficient.'),
        DeclareLaunchArgument(
            'dpso_w_goal',
            default_value=str(dpso_swarm.DPSOParams.w_goal),
            description='DPSO (method_3/6): J weight on running goal distance.'),
        DeclareLaunchArgument(
            'dpso_w_ob',
            default_value=str(dpso_swarm.DPSOParams.w_ob),
            description='DPSO (method_3/6): J weight on neighbour keep-out.'),
        DeclareLaunchArgument(
            'dpso_w_effort',
            default_value=str(dpso_swarm.DPSOParams.w_effort),
            description='DPSO (method_3/6): J weight on |v|^2 effort.'),
        DeclareLaunchArgument(
            'dpso_w_smooth',
            default_value=str(dpso_swarm.DPSOParams.w_smooth),
            description='DPSO (method_3/6): J weight on |dv|^2 smoothness.'),
        DeclareLaunchArgument(
            'dpso_keepout',
            default_value=str(dpso_swarm.DPSOParams.keepout),
            description='DPSO (method_3/6): neighbour keep-out radius in J.'),
        DeclareLaunchArgument(
            'dpso_ob_stop',
            default_value=str(dpso_swarm.DPSOParams.ob_stop),
            description='DPSO (method_3/6): neighbour penalty outer range.'),
        DeclareLaunchArgument(
            'dpso_speed_cap',
            default_value=str(dpso_swarm.DPSOParams.speed_cap),
            description='DPSO (method_3/6): velocity magnitude cap (m/s).'),
        DeclareLaunchArgument(
            'hull_activation',
            default_value=str(convex_hull_avoid.HullParams.activation),
            description='Convex hull (method_4): outside-hull push range (m).'),
        DeclareLaunchArgument(
            'hull_gain',
            default_value=str(convex_hull_avoid.HullParams.gain),
            description='Convex hull (method_4): edge-push gain (1/s).'),
        DeclareLaunchArgument(
            'hull_base_speed',
            default_value=str(convex_hull_avoid.HullParams.base_speed),
            description='Convex hull (method_4): toward-goal cruise speed (m/s).'),
        DeclareLaunchArgument(
            'hull_speed_cap',
            default_value=str(convex_hull_avoid.HullParams.speed_cap),
            description='Convex hull (method_4): reference magnitude cap (m/s).'),
        DeclareLaunchArgument(
            'gmm_sigma',
            default_value=str(gmm_avoid.GMMParams.sigma),
            description='GMM (method_5/6): Gaussian kernel width (m).'),
        DeclareLaunchArgument(
            'gmm_gain',
            default_value=str(gmm_avoid.GMMParams.gain),
            description='GMM (method_5): density-repulsion gain (1/s).'),
        DeclareLaunchArgument(
            'gmm_activation',
            default_value=str(gmm_avoid.GMMParams.activation),
            description='GMM (method_5): density-push activation range (m).'),
        DeclareLaunchArgument(
            'gmm_base_speed',
            default_value=str(gmm_avoid.GMMParams.base_speed),
            description='GMM (method_5): toward-goal cruise speed (m/s).'),
        DeclareLaunchArgument(
            'gmm_speed_cap',
            default_value=str(gmm_avoid.GMMParams.speed_cap),
            description='GMM (method_5): reference magnitude cap (m/s).'),
        DeclareLaunchArgument(
            'comb_w_gmm',
            default_value=str(combined_avoid.CombinedParams.comb_w_gmm),
            description='Combined (method_6): J weight of the GMM potential.'),
        DeclareLaunchArgument(
            'stgo_adoption',
            default_value=str(combined_avoid.CombinedParams.stgo_adoption),
            description='Combined (method_6): STGO adoption weight '
                        '(1.0 = follow the swarm-optimal plan, '
                        '0.0 = keep the local plan).'),

        DeclareLaunchArgument(
            'render_engine',
            default_value='ogre',
            choices=['ogre2', 'ogre', 'optix', 'headless'],
            description='Gazebo Harmonic render engine. ogre is the reliable '
                        'default on Wayland/limited-display setups; headless '
                        'runs server-only with no GUI window and lifts the '
                        'real-time physics lock (turbo: fly the scenario as '
                        'fast as the CPU allows).'),

        DeclareLaunchArgument(
            'physics_engine',
            default_value='ode',
            choices=['ode', 'tpe', 'bullet', 'dart'],
            description='gz-physics engine plugin used by the gz-sim <-z> '
                        'transient world. ode is the default and the only '
                        'engine verified with the multicopter control system.'),

        DeclareLaunchArgument(
            'drone_count',
            default_value='2',
            choices=['2', '5', '10'],
            description='Number of drones to generate and spawn this session.'),

        DeclareLaunchArgument(
            'lane_offset',
            default_value='0.0',
            description='Break the symmetric single-point crossing: alternate '
                        'drones\'y=5 target x are shifted +/- lane_offset '
                        'metres (0 = every path crosses the exactly same '
                        'point, the degenerate worst case).'),

        DeclareLaunchArgument(
            'drone_model',
            default_value='full',
            choices=['full', 'lite'],
            description="'full' keeps the bundled Bebop camera + GPU-lidar; "
                        "'lite' overlays the same airframe without the "
                        'GPS-unused sensors to cut sim load at scale.'),

        DeclareLaunchArgument(
            'altitude_lane',
            default_value='0.3',
            description='Target-z altitude ladder spacing in m for N >= 5 '
                        '(0 disables). Spreads the mirrored single-point '
                        'crossing vertically so dense flocks clear it.'),

        DeclareLaunchArgument(
            'drone_size_kg',
            default_value='0.5',
            choices=['0.5', '5.0', '20.0'],
            description='Drone mass in kg. 0.5 is the real Parrot Bebop 2 '
                        'model; 5.0/20.0 rewrite the model with the physics '
                        'of a real airframe of that weight class (DJI Matrice '
                        '300 RTK class quad / DJI Agras T10 class field '
                        'quad: real mass, inertia, wheelbase rotor geometry, '
                        'rotor constants and hover RPM).'),

        DeclareLaunchArgument(
            'wind',
            default_value='off',
            choices=['off',
                     'strong_1', 'strong_3', 'strong_5', 'strong_10',
                     'gust_1', 'gust_3', 'gust_5', 'gust_10'],
            description='Wind via the WindEffects field + native gz_aero_drag '
                        'per-mass drag. Each non-off value encodes mode + '
                        'speed in m/s along +x: strong_N is a constant N m/s '
                        'cross-wind, gust_N adds a sinusoidal gust. Flight '
                        'envelope: 1-3 m/s fine for every drone, 5 m/s is '
                        'held by every airframe at a small steady downwind '
                        'offset (gust modes add excursions); 10 m/s exceeds '
                        'the command envelope (documented limit).'),

        DeclareLaunchArgument(
            'seed',
            default_value='auto',
            description='gz-sim transient world seed. "auto" picks a fresh '
                        'random seed per launch so stale state cannot leak '
                        'between sessions.'),

        DeclareLaunchArgument(
            'show_labels',
            default_value='true',
            choices=['true', 'false'],
            description='Show a TEXT marker above each drone with its name '
                        'and target colour (requires gz-sim-markers-system '
                        'plugin in the world).'),

        DeclareLaunchArgument(
            'goal_timeout',
            default_value='',
            description='Seconds to wait for every drone to report '
                        'goal_reached before the start gate closes the '
                        'simulation and reports the timeout as the elapsed '
                        'maneuver time. Leave empty for the wind-aware '
                        'default: 300 s with wind off / 600 s in wind '
                        'modes; an explicit value always wins.'),

        DeclareLaunchArgument(
            'collision_radius',
            default_value='0.7',
            description='Horizontal collision-detection radius in metres; a '
                        'drone reports <prefix>/collision when a neighbour '
                        'centre is within this distance horizontally AND '
                        'collision_vertical metres vertically. Override '
                        'larger than the 2.0 m spawn spacing to force a '
                        'detection for testing the counting chain.'),

        OpaqueFunction(function=_launch_setup),
    ])