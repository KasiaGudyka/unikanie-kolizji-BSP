#!/usr/bin/env python3

"""Benchmark the collision-avoidance methods under a run matrix.

The run matrix is the Cartesian product of ``method x drone_count x mass``,
each repeated ``--repeats`` times (default 30) with a fixed per-repeat seed, all
in headless mode (turbo) and with a shared ``--goal-timeout`` (default 120 s).

For every run four independent, dependency-free measurement channels are
gathered:

  * per-tick wall/CPU time of the avoidance node — the ``perf_probe`` csv/json
    written by ``drone_collision_avoidance`` (gated, default off; the csv is
    kill-safe, the json also carries per-drone flown path ``path_max`` m);
  * Gazebo real-time factor — ``/world/empty_world_harmonic/stats``
    (``gz.msgs10.WorldStatistics``) sampled at 1 Hz;
  * per-process CPU % (gz-sim server and the avoidance node) — jiffie deltas
    from ``/proc/<pid>/stat``;
  * an optional Intel-iGPU busy proxy — ``rc6_residency_ms`` delta plus the
    ``rps_cur/rps_max_freq_mhz`` ratio from sysfs.

Mission outcome (elapsed seconds, goal_reached flag, collisions) comes from the
start-gate's ``/simulation/elapsed``, ``/simulation/goal_reached`` and
``/simulation/collisions`` topics.  A run is:

  * ``success``      — every drone reached its target (elapsed < goal_timeout);
  * ``goal_timeout`` — the gate's goal deadline was reached (drone(s) stuck);
  * ``aborted``      — no ``elapsed`` within ``--run-timeout`` (upstream fail).

Outputs under ``--out``: per-run rows in ``bench_runs.csv`` (append-only and
resumable — already-finished ``(count, mass, method, rep)`` rows are skipped on
restart) and the analysis table ``bench_matrix.csv`` with one row per
combination, holding per the plan: average maneuver time, average time of the
successful runs, average wall-clock time, timeout %, runs-with-collisions %,
max/min collisions and longest/shortest per-drone path, plus the perf/CPU
extras (tick p95, 20 Hz rate, gz/avoid CPU %, RTF).

No external tools are installed or required.
"""

import argparse
import csv
import glob
import json
import os
import signal
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone

import gz.msgs10.world_stats_pb2 as world_stats_pb2
import gz.transport13 as gz
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Bool, Float64, Int32

from unikanie_kolizji.drone_gps import ELAPSED_TOPIC, VIEW_READY_TOPIC
from unikanie_kolizji.drone_start_gate import COLLISIONS_TOPIC, GOAL_REACHED_TOPIC

WORLD = 'empty_world_harmonic'
STATS_TOPIC = f'/world/{WORLD}/stats'
CLK_HZ = os.sysconf('SC_CLK_TCK')

RC6_GLOB = '/sys/class/drm/card*/device/gt/gt0/rc6_residency_ms'
RPS_CUR = '/sys/class/drm/card*/device/gt/gt0/rps_cur_freq_mhz'
RPS_MAX = '/sys/class/drm/card*/device/gt/gt0/rps_max_freq_mhz'

METHODS = ['method_1', 'method_2', 'method_3', 'method_4', 'method_5', 'method_6']
COUNTS = [2, 5, 10]
MASSES = [0.5, 5.0, 20.0]


# --- small helpers -----------------------------------------------------------

def _read(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def _pids_for(*substrs):
    """PIDs whose cmdline contains every ``substrs`` (empty if none)."""
    found = []
    for entry in glob.glob('/proc/[0-9]*/cmdline'):
        try:
            cmd = open(entry, 'rb').read().replace(b'\0', b' ').decode(
                'utf-8', 'replace')
        except OSError:
            continue
        if all(s in cmd for s in substrs):
            found.append(int(entry.split('/')[2]))
    return found


def _jiffies(pid):
    """utime+stime jiffies for ``pid`` from /proc stat (None on error)."""
    try:
        with open(f'/proc/{pid}/stat') as f:
            fields = f.read().split()
        return int(fields[13]) + int(fields[14])
    except (OSError, ValueError, IndexError):
        return None


# --- collectors --------------------------------------------------------------

class RWStats:
    """Thread-safe holders for all collector outputs of one run."""

    def __init__(self):
        self.rtf = []
        self.iterations = 0
        self.elapsed = None          # seconds (goal reach or timeout)
        self.goal_reached = None     # True/False when the maneuver finished
        self.collisions = None
        self.gz_cpu = None           # mean %CPU of gz-sim server
        self.avoid_cpu = None        # mean %CPU of the avoidance node
        self.rc6_busy = None         # 1 - rc6 idle fraction (0..1), Intel
        self.rps_ratio = None        # cur/max GT frequency at run end
        self.probe = None            # perf_<method>.json dict
        self.wall = None
        self.rtf_mean = self.rtf_p5 = self.rtf_min = None


class CPUSampler2(threading.Thread):
    """Cumulative %CPU for a set of PIDs across the whole run."""

    def __init__(self, pids, tag, out):
        super().__init__(daemon=True)
        self.pids = list(pids)
        self.tag = tag
        self.out = out
        self.stop_evt = threading.Event()

    def stop(self):
        self.stop_evt.set()

    def run(self):
        if not self.pids:
            setattr(self.out, self.tag, 0.0)
            return
        prev = {p: _jiffies(p) for p in self.pids if _jiffies(p) is not None}
        t0 = time.monotonic()
        while not self.stop_evt.wait(0.5):
            cur = {p: _jiffies(p) for p in self.pids if _jiffies(p) is not None}
            t1 = time.monotonic()
            if cur and prev:
                d_j = sum(cur[p] - prev[p] for p in cur if p in prev)
                d_t = t1 - t0
                setattr(self.out, self.tag, 100.0 * d_j / CLK_HZ / d_t)
            prev = cur
            t0 = t1


class RCLOut(Node):
    """ROS subscribers latching the mission outcome of the current run."""

    def __init__(self, out):
        super().__init__('benchmark_avoidance')
        self.out = out
        self.create_subscription(Float64, ELAPSED_TOPIC,
                                 self._elapsed_cb, 10)
        self.create_subscription(Bool, GOAL_REACHED_TOPIC,
                                 self._goal_reached_cb, 10)
        self.create_subscription(Int32, COLLISIONS_TOPIC,
                                 self._collision_cb, 10)
        # Headless runs have no GUI to latch /simulation/view_ready; the start
        # gate would otherwise idle for view_ready_timeout (60 s) before the
        # maneuver begins. Latch it ourselves so the mission starts promptly.
        qos = self._latching_qos()
        self._view_ready_pub = self.create_publisher(
            Bool, VIEW_READY_TOPIC, qos)
        self._sent_view_ready = False

    @staticmethod
    def _latching_qos():
        from rclpy.qos import DurabilityPolicy, QoSProfile
        profile = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        return profile

    def _elapsed_cb(self, msg):
        self.out.elapsed = msg.data

    def _goal_reached_cb(self, msg):
        self.out.goal_reached = bool(msg.data)

    def _collision_cb(self, msg):
        self.out.collisions = msg.data

    def latch_view_ready(self):
        """Keep re-publishing the latch briefly so the gate reliably sees it."""
        if self._sent_view_ready:
            return
        deadline = time.monotonic() + 5.0
        while rclpy.ok() and time.monotonic() < deadline:
            self._view_ready_pub.publish(Bool(data=True))
            time.sleep(0.5)
        self._sent_view_ready = True


# --- one run -----------------------------------------------------------------

def _outcome(out, goal_timeout):
    if out.goal_reached is True:
        return 'success'
    if out.goal_reached is False:
        return 'goal_timeout'
    if out.elapsed is None:
        return 'aborted'
    # Fallback for a gate without the flag: goal_timeout == elapsed value.
    return 'goal_timeout' if out.elapsed >= goal_timeout else 'success'


def _run_once(method, count, mass, rep, outdir, args, seed):
    out = RWStats()
    perf_dir = os.path.join(outdir, 'probe', f'c{count}', f'm{mass:g}', method)
    overrides = [
        f'render_engine:={args.render_engine}',
        f'collision_avoidance_method:={method}',
        'recording_enabled:=false',
        'perf_probe:=true',
        f'perf_dir:={perf_dir}',
        f'drone_count:={count}',
        f'drone_size_kg:={mass:.1f}',
        f'goal_timeout:={args.goal_timeout:g}',
    ]
    if not any(x.startswith('seed:=') for x in args.extra):
        overrides += [f'seed:={seed}']
    overrides += args.extra

    cmd = [args.ros2, 'launch', 'unikanie_kolizji',
           'gazebo_two_drones.launch.py'] + overrides

    # ROS side first: latch elapsed/collisions before the mission can finish.
    rclpy.init()
    rcl = RCLOut(out) if rclpy.ok() else None
    executor = SingleThreadedExecutor() if rcl else None
    ros_thread = None
    if executor:
        executor.add_node(rcl)
        ros_thread = threading.Thread(target=executor.spin, daemon=True)
        ros_thread.start()

    # gz side: sample the world stats topic.
    gz_node = gz.Node()

    def _stats_cb(msg):
        out.rtf.append(msg.real_time_factor)
        out.iterations = msg.iterations

    gz_node.subscribe(world_stats_pb2.WorldStatistics, STATS_TOPIC,
                      _stats_cb)

    rc6_pat = glob.glob(RC6_GLOB)
    rc6_start = int(_read(rc6_pat[0])) if rc6_pat else None
    t_start = time.monotonic()

    proc = subprocess.Popen(cmd, start_new_session=True,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            cwd=args.cwd)
    pgid = os.getpgid(proc.pid)
    if rcl is not None:
        threading.Thread(target=rcl.latch_view_ready, daemon=True).start()
    gz_sampler = avoid_sampler = None

    try:
        gz_pids, avoid_pids = [], []
        seed_arg = f'-z {seed}'
        deadline = time.monotonic() + args.startup_timeout
        while time.monotonic() < deadline:
            gz_pids = _pids_for('gz', 'sim', '--headless-rendering',
                                seed_arg)
            if not gz_pids:
                gz_pids = _pids_for('gz-sim-server', seed_arg)
            avoid_pids = _pids_for('drone_collision_avoidance')
            if gz_pids and avoid_pids:
                break
            time.sleep(0.5)

        gz_sampler = CPUSampler2(gz_pids, 'gz_cpu', out)
        avoid_sampler = CPUSampler2(avoid_pids, 'avoid_cpu', out)
        gz_sampler.start()
        avoid_sampler.start()

        deadline = time.monotonic() + args.run_timeout
        while time.monotonic() < deadline:
            if out.elapsed is not None:
                break
            time.sleep(0.5)
    finally:
        try:
            os.killpg(pgid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
        time.sleep(3.0)
        try:
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        # gz may have escaped the launch session; finish off any server
        # that still runs this run's seed so no orphans are left behind.
        for zombie in _pids_for('gz', 'sim', '--headless-rendering',
                                seed_arg):
            try:
                os.kill(zombie, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        # allow the on_shutdown perf dump + kill-safe csv flush to finish.
        time.sleep(1.0)
        if gz_sampler is not None:
            gz_sampler.stop(); gz_sampler.join(timeout=4)
        if avoid_sampler is not None:
            avoid_sampler.stop(); avoid_sampler.join(timeout=4)

    try:
        wall = time.monotonic() - t_start
        out.wall = wall

        # rc6 / rps GPU proxy (late sample after the heavy part).
        rc6_end = int(_read(rc6_pat[0])) if rc6_pat else None
        if rc6_start is not None and rc6_end is not None and wall > 5.0:
            idle = max(0.0, min(1.0, (rc6_end - rc6_start) / (wall * 1e3)))
            out.rc6_busy = 1.0 - idle
        cur_f = _read(glob.glob(RPS_CUR)[0]) if glob.glob(RPS_CUR) else None
        max_f = _read(glob.glob(RPS_MAX)[0]) if glob.glob(RPS_MAX) else None
        if cur_f and max_f:
            out.rps_ratio = int(cur_f) / int(max_f)

        # perf probe summary: prefer the node's JSON, fall back to its CSV.
        probe_json = os.path.join(perf_dir, f'perf_{method}.json')
        probe_csv = os.path.join(perf_dir, f'perf_{method}.csv')
        out.probe = _load_perf(probe_json, probe_csv)

        out.rtf_mean = statistics.mean(out.rtf) if out.rtf else None
        out.rtf_p5 = _pct(out.rtf, 5) if out.rtf else None
        out.rtf_min = min(out.rtf) if out.rtf else None
    finally:
        if executor is not None:
            executor.shutdown()
        if ros_thread is not None:
            ros_thread.join(timeout=2)
        if rcl is not None:
            rcl.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    return out


def _load_perf(json_path, csv_path):
    """Return the probe dict, reading json or computing it from the csv."""
    if os.path.exists(json_path):
        try:
            with open(json_path) as f:
                return json.load(f)
        except (OSError, ValueError):
            pass
    if not os.path.exists(csv_path):
        return None
    wall, cpu, dt = [], [], []
    try:
        with open(csv_path) as f:
            next(f, None)  # header
            for line in f:
                parts = line.split(',')
                if len(parts) < 3:
                    continue
                wall.append(int(parts[0]))
                cpu.append(int(parts[1]))
                dt.append(float(parts[2]))
    except (OSError, ValueError):
        return None
    if not wall:
        return None
    return {
        'n': len(wall),
        'wall_ms': {'mean': statistics.mean(wall) / 1e6,
                    'p50': _pct(wall, 50) / 1e6,
                    'p95': _pct(wall, 95) / 1e6,
                    'max': max(wall) / 1e6},
        'cpu_ms': {'mean': statistics.mean(cpu) / 1e6,
                   'p50': _pct(cpu, 50) / 1e6,
                   'p95': _pct(cpu, 95) / 1e6,
                   'max': max(cpu) / 1e6},
        'interval_s': {'mean': statistics.mean(dt)},
        'rate_hz': {'mean': 1.0 / statistics.mean(dt) if dt else None},
    }


def _pct(values, p):
    if not values:
        return None
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, int(round(p / 100 * (len(ordered) - 1)))))
    return ordered[idx]


# --- rows --------------------------------------------------------------------

def _row(out, method, count, mass, rep, seed, goal_timeout):
    probe = out.probe or {}
    wall_ms = probe.get('wall_ms') or {}
    cpu_ms = probe.get('cpu_ms') or {}
    rate = (probe.get('rate_hz') or {}).get('mean')
    pm = probe.get('path_m')
    path_max = max(pm.values()) if pm else probe.get('path_max')
    return {
        'method': method, 'count': count, 'mass': mass,
        'rep': rep, 'seed': seed,
        'outcome': _outcome(out, goal_timeout),
        'goal_reached': out.goal_reached,
        'tick_wall_p50_ms': wall_ms.get('p50'),
        'tick_wall_p95_ms': wall_ms.get('p95'),
        'tick_wall_mean_ms': wall_ms.get('mean'),
        'tick_cpu_p50_ms': cpu_ms.get('p50'),
        'tick_cpu_p95_ms': cpu_ms.get('p95'),
        'tick_rate_hz': rate,
        'n_ticks': probe.get('n'),
        'rtf_mean': out.rtf_mean, 'rtf_p5': out.rtf_p5,
        'rtf_min': out.rtf_min,
        'gz_cpu_pct': out.gz_cpu, 'avoid_cpu_pct': out.avoid_cpu,
        'gpu_rc6_busy': out.rc6_busy, 'gpu_rps_ratio': out.rps_ratio,
        'elapsed_s': out.elapsed, 'collisions': out.collisions,
        'path_max_m': path_max,
        'wall_s': out.wall,
    }


def _matrix_row(rows, method, count, mass, goal_timeout):
    rs = [r for r in rows if (r['method'] == method and r['count'] == count
                              and r['mass'] == mass)]
    if not rs:
        return None
    n = len(rs)
    ended = [r['elapsed_s'] for r in rs if r['elapsed_s'] is not None]
    success = [r['elapsed_s'] for r in rs if r['outcome'] == 'success'
               and r['elapsed_s'] is not None]
    timeouts = sum(1 for r in rs if r['outcome'] in ('goal_timeout',
                                                     'aborted'))
    colls = [c for c in (r['collisions'] for r in rs) if c is not None]
    withcoll = sum(1 for c in colls if c > 0)
    paths = [r['path_max_m'] for r in rs if r['path_max_m'] is not None]

    def _mean(vals):
        vals = [v for v in vals if v is not None]
        return statistics.mean(vals) if vals else None

    tick_p95 = [r['tick_wall_p95_ms'] for r in rs if r['tick_wall_p95_ms']]
    rate = [r['tick_rate_hz'] for r in rs if r['tick_rate_hz']]
    gz_cpu = [r['gz_cpu_pct'] for r in rs if r['gz_cpu_pct']]
    avoid_cpu = [r['avoid_cpu_pct'] for r in rs if r['avoid_cpu_pct']]
    rtf = [r['rtf_mean'] for r in rs if r['rtf_mean']]
    wall = [r['wall_s'] for r in rs if r['wall_s']]

    return {
        'count': count, 'mass': mass, 'method': method, 'n_runs': n,
        'time_mean_s': _mean(ended),
        'time_success_mean_s': _mean(success),
        'wall_mean_s': _mean(wall),
        'timeout_pct': round(100.0 * timeouts / n, 2),
        'coll_pct': round(100.0 * withcoll / n, 2),
        'coll_max': max(colls) if colls else None,
        'coll_min': min(colls) if colls else None,
        'path_longest_m': max(paths) if paths else None,
        'path_shortest_m': min(paths) if paths else None,
        'path_mean_m': _mean(paths),
        'tick_p95_mean_ms': _mean(tick_p95),
        'tick_rate_hz_mean': _mean(rate),
        'gz_cpu_mean_pct': _mean(gz_cpu),
        'avoid_cpu_mean_pct': _mean(avoid_cpu),
        'rtf_mean': _mean(rtf),
    }


def _fmt(v, nd=3):
    return f'{v:.{nd}f}' if isinstance(v, float) else ('' if v is None else v)


def _load_existing(run_csv):
    """Return (rows, done_keys) already recorded in ``run_csv``."""
    if not os.path.exists(run_csv):
        return [], set()
    rows = []
    done = set()
    with open(run_csv, newline='') as f:
        for r in csv.DictReader(f):
            rows.append(r)
            done.add((int(r['count']), float(r['mass']), r['method'],
                      int(r['rep'])))
    return rows, done


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--methods', nargs='*', default=METHODS,
                        choices=METHODS,
                        help='collision-avoidance methods (default: all 6).')
    parser.add_argument('--counts', nargs='*', type=int, default=[5, 10],
                        choices=COUNTS,
                        help='drone counts to benchmark (default: 5 10).')
    parser.add_argument('--masses', nargs='*', type=float, default=[5.0, 20.0],
                        choices=MASSES,
                        help='drone masses in kg (default: 5.0 20.0).')
    parser.add_argument('--repeats', type=int, default=30,
                        help='repeats per combination with a fresh seed '
                             '(default 30).')
    parser.add_argument('--goal-timeout', type=float, default=120.0,
                        help='gate goal deadline in sim-seconds (default 120); '
                             'elapsed == this value marks a timed-out run.')
    parser.add_argument('--out', default='', help='output directory '
                        '(default ~/.gz/sim/log/bench_<ts>/).')
    parser.add_argument('--render-engine', default='headless',
                        choices=['headless', 'ogre', 'ogre2'],
                        help='gz render engine (default headless = turbo).')
    parser.add_argument('--startup-timeout', type=float, default=90.0,
                        help='seconds to wait for gz-server + avoidance '
                             'processes to appear.')
    parser.add_argument('--run-timeout', type=float, default=600.0,
                        help='seconds to wait for /simulation/elapsed before '
                             'tearing the run down.')
    parser.add_argument('--base-seed', type=int, default=20260923,
                        help='seed offset; repeat r uses base_seed + r.')
    parser.add_argument('--ros2', default='ros2',
                        help='ros2 launcher binary (default: ros2 on PATH).')
    parser.add_argument('--cwd', default='.', help='launch working directory.')
    parser.add_argument('--extra', action='append', default=[],
                        metavar='name:=value',
                        help='extra launch argument, repeatable.')
    args = parser.parse_args(argv)

    outdir = args.out or os.path.join(
        os.path.expanduser('~'), '.gz', 'sim', 'log',
        'bench_' + datetime.now(timezone.utc).strftime('%Y-%m-%dT%H-%M-%S'))
    os.makedirs(outdir, exist_ok=True)
    run_csv = os.path.join(outdir, 'bench_runs.csv')
    matrix_csv = os.path.join(outdir, 'bench_matrix.csv')
    matrix_headers = ['count', 'mass', 'method', 'n_runs', 'time_mean_s',
                      'time_success_mean_s', 'wall_mean_s', 'timeout_pct',
                      'coll_pct', 'coll_max', 'coll_min',
                      'path_longest_m', 'path_shortest_m', 'path_mean_m',
                      'tick_p95_mean_ms', 'tick_rate_hz_mean',
                      'gz_cpu_mean_pct', 'avoid_cpu_mean_pct', 'rtf_mean']

    rows, done = _load_existing(run_csv)
    fieldnames = list(_row(RWStats(), 'method_1', 5, 5.0, 1, 0,
                           args.goal_timeout).keys())
    combos = [(c, m, method)
              for c in args.counts for m in args.masses
              for method in args.methods]
    total = len(combos) * args.repeats
    print(f'Matrix: {len(args.methods)} methods x '
          f'{args.counts} drones x {args.masses} kg x '
          f'{args.repeats} reps = {total} runs')
    print(f'  out -> {outdir}  (resume: {len(done)} already recorded)')

    with open(run_csv, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not rows:
            writer.writeheader()
        t0 = time.monotonic()
        n_done = len(done)
        for count, mass, method in combos:
            for rep in range(1, args.repeats + 1):
                seed = args.base_seed + rep
                key = (count, mass, method, rep)
                if key in done:
                    continue
                print(f'-- {method} c{count} m{mass:g} rep '
                      f'{rep}/{args.repeats} seed={seed}')
                out = _run_once(method, count, mass, rep, outdir, args, seed)
                row = _row(out, method, count, mass, rep, seed,
                           args.goal_timeout)
                rows.append(row)
                writer.writerow(row)
                f.flush()
                done.add(key)
                n_done += 1
                dt = time.monotonic() - t0
                eta = dt / n_done * (total - n_done)
                print(f'   {row["outcome"]:11s} elapsed='
                      f'{_fmt(row["elapsed_s"])}s coll='
                      f'{_fmt(row["collisions"], 0)} path='
                      f'{_fmt(row["path_max_m"])}m wall={row["wall_s"]:.0f}s '
                      f'[{n_done}/{total} {dt / n_done:.0f}s/run '
                      f'ETA {eta / 60:.0f}min]', flush=True)
                _write_matrix(matrix_csv, matrix_headers, rows, args.methods,
                              args.counts, args.masses, args.goal_timeout)

    _write_matrix(matrix_csv, matrix_headers, rows, args.methods, args.counts,
                  args.masses, args.goal_timeout, verbose=True)
    print(f'\nPer-run rows  -> {run_csv}')
    print(f'Matrix table  -> {matrix_csv}')
    return 0


def _write_matrix(path, headers, rows, methods, counts, masses, goal_timeout,
                  verbose=False):
    """(Re)write the per-combination summary table from ``rows``."""
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for count in counts:
            for mass in masses:
                for method in methods:
                    row = _matrix_row(rows, method, count, mass, goal_timeout)
                    if row is not None:
                        writer.writerow(row)
    if verbose:
        print('\nPer-combination summary:')
        print('  cn  mass | method | n | avgT | avgTsuc | wall | tmout% | '
              'coll% | cMax cMin | pLong pShort | tickP95 | rateHz | '
              'gzCPU avoid rtf')
        with open(path) as f:
            for r in csv.DictReader(f):
                cn = int(r['count'])
                print(f"  {cn:>3d} {r['mass']:>4s} | "
                      f"{r['method']:9s} | {r['n_runs']:>2s} | "
                      f"{_fmt(r['time_mean_s']):>6s} | "
                      f"{_fmt(r['time_success_mean_s']):>7s} | "
                      f"{_fmt(r['wall_mean_s']):>5s} | "
                      f"{r['timeout_pct']:>6s} | "
                      f"{r['coll_pct']:>6s} | "
                      f"{_fmt(r['coll_max'], 0):>4s} {_fmt(r['coll_min'], 0):>4s} | "
                      f"{_fmt(r['path_longest_m']):>6s} {_fmt(r['path_shortest_m']):>7s} | "
                      f"{_fmt(r['tick_p95_mean_ms']):>9s} | "
                      f"{_fmt(r['tick_rate_hz_mean']):>7s} | "
                      f"{_fmt(r['gz_cpu_mean_pct']):>6s} "
                      f"{_fmt(r['avoid_cpu_mean_pct']):>6s} "
                      f"{_fmt(r['rtf_mean']):>5s}")


if __name__ == '__main__':
    sys.exit(main())