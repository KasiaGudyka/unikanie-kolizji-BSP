#!/usr/bin/env python3

"""Global start gate for the drone swarm.

Each ``drone_gps`` node publishes a ``ready`` Bool on ``/<prefix>/ready`` once
its model has been stable at its start pose for several ticks.  This node
subscribes to every drone's ready topic and, when ALL drones are ready,
publishes ``/simulation/go`` so the whole swarm enters the maneuver at the same
time.  Without this gate a drone that is ready early would already be flying
while a slower-loading neighbour is still settling, which is exactly the
out-of-position spawn / stray-trajectory problem the readiness handshake is
meant to eliminate.

A ``ready_timeout`` fails open: if one drone gets stuck before it can confirm
its start pose, the gate still releases the swarm after the timeout (with a
loud warning) instead of hanging the session forever.

The gate also owns the maneuver timer: it records the moment it publishes ``go``
(all drones ready -> timer starts) and, once every drone has reported
``goal_reached``, publishes the elapsed seconds on ``/simulation/elapsed``.  It
also collects every drone's ``/<prefix>/collision`` Bool, latches the set of
drones that ever collided and publishes the number of collisions (distinct
drones / 2) on ``/simulation/collisions``.
A ``goal_timeout`` covers a drone that never reaches its target: after the
timeout the gate publishes the configured timeout as the result and terminates
the simulation (gz sim has no shutdown control, so it best-effort kills the
session's gz processes; the launch's OnProcessExit tears the stack down).
"""

import math
import os
import signal
import struct
import time

import subprocess
from datetime import datetime, timezone

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from geometry_msgs.msg import Twist
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import Bool, Float64, Int32

from unikanie_kolizji.drone_gps import (ELAPSED_TOPIC, GO_TOPIC,
                                    VIEW_READY_TOPIC)

# Aggregate collision counter topic: std_msgs/Int32 with the number of distinct
# collisions observed this run (involved drones / 2). Published whenever the
# count changes and again when the maneuver finishes.
COLLISIONS_TOPIC = '/simulation/collisions'

# Outcome flag topic: std_msgs/Bool published when the maneuver ends — True
# when every drone reached its target, False when the goal_timeout fired.
GOAL_REACHED_TOPIC = '/simulation/goal_reached'


def should_start(ready: dict, prefixes: list, elapsed: float,
                 timeout: float) -> bool:
    """True when every drone has reported ready, or the timeout has expired.

    ``ready`` maps prefix -> bool.  ``elapsed`` is the seconds since the node
    started its check loop.  Unless ``elapsed`` exceeded ``timeout`` (fail-open)
    the whole swarm must be ready before the gate opens.
    """
    if elapsed >= timeout:
        return True
    return all(ready.get(p, False) for p in prefixes)


def all_goals(goals: dict, prefixes: list) -> bool:
    """True when every drone has reported goal_reached."""
    return all(goals.get(p, False) for p in prefixes)


def collision_count(collided: dict) -> int:
    """Distinct collisions observed so far.

    Each physical collision is reported by exactly the two drones involved, so
    the number of collisions is the number of drones that latched a collision
    report, divided by two (integer).
    """
    return sum(1 for v in collided.values() if v) // 2


def elapsed_seconds(go_ns: int, now_ns: int) -> float:
    """Seconds between the go timestamp and ``now`` (>= 0)."""
    return max(0.0, (now_ns - go_ns) / 1e9)


class _TlogFrameWriter:
    """MAVProxy-style tlog sink: an 8-byte BE microsecond timestamp + frame.

    plot.ardupilot.org's parser (``mavlink.js`` ``parseType``) reads the 8
    bytes directly preceding every MAVLink frame as a GCS reception timestamp
    and silently drops any message whose derived ``time_boot_ms`` exceeds 100
    days.  Bare pymavlink output (frames back-to-back, no prefix) makes that
    parser see the previous frame's tail as the timestamp and throw away every
    ATTITUDE, which is the site's hard map gate.  This sink prepends each real
    frame with ``struct.pack('>Q', ts)`` so the site keeps every message and
    plots a sensible time axis.  ``ts_host`` is a 1-element list of the current
    microseconds-since-boot, updated by the recorder before each send.
    """

    def __init__(self, fh, ts_host):
        self._fh = fh
        self._ts_host = ts_host
        self.name = fh.name

    def write(self, buf):
        if buf and buf[0] in (253, 254):  # MAVLink1/2 magic: timestamp + frame
            self._fh.write(struct.pack('>Q', self._ts_host[0]))
            self._fh.write(buf)
        else:
            self._fh.write(buf)

    def flush(self):
        self._fh.flush()

    def close(self):
        self._fh.close()


class DroneStartGateNode(Node):
    def __init__(self):
        super().__init__('drone_start_gate')

        self.declare_parameter('drone_prefixes', ['drone_0', 'drone_1'])
        self.declare_parameter('ready_timeout', 30.0)
        self.declare_parameter('goal_timeout', 300.0)
        self.declare_parameter('check_interval', 0.1)
        self.declare_parameter('view_ready_timeout', 60.0)
        self.declare_parameter('recording_enabled', True)

        self.prefixes = list(self.get_parameter('drone_prefixes')
                             .get_parameter_value().string_array_value)
        self.timeout = (self.get_parameter('ready_timeout')
                        .get_parameter_value().double_value)
        self.goal_timeout = (self.get_parameter('goal_timeout')
                             .get_parameter_value().double_value)
        self.interval = (self.get_parameter('check_interval')
                         .get_parameter_value().double_value)
        self.view_ready_timeout = (self.get_parameter('view_ready_timeout')
                                   .get_parameter_value().double_value)
        self.recording_enabled = (self.get_parameter('recording_enabled')
                                  .get_parameter_value().bool_value)

        transient = QoSProfile(depth=10,
                               durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._go_pub = self.create_publisher(Bool, GO_TOPIC, transient)
        self._ready = {prefix: False for prefix in self.prefixes}
        self._view_ready = False
        self._goal = {prefix: False for prefix in self.prefixes}
        self._collided = {prefix: False for prefix in self.prefixes}
        self._last_collisions = -1
        self._subs = []
        self._subs.append(self.create_subscription(
            Bool, VIEW_READY_TOPIC,
            lambda msg: setattr(self, '_view_ready', msg.data), transient))
        for prefix in self.prefixes:
            self._subs.append(self.create_subscription(
                Bool, f'/{prefix}/ready',
                lambda msg, p=prefix: self._ready_cb(p, msg), transient))
            self._subs.append(self.create_subscription(
                Bool, f'/{prefix}/goal_reached',
                lambda msg, p=prefix: self._goal_cb(p, msg), transient))
            self._subs.append(self.create_subscription(
                Bool, f'/{prefix}/collision',
                lambda msg, p=prefix: self._collision_cb(p, msg), transient))

        # Elapsed time: volatile QoS so a previous run's value can never stick
        # to a fresh run when the subscriber is created.
        volatile = QoSProfile(depth=10)
        self._elapsed_pub = self.create_publisher(Float64, ELAPSED_TOPIC,
                                                  volatile)
        # Aggregate collision counter: volatile, republished whenever the
        # latch changes and at the maneuver finish.
        self._collisions_pub = self.create_publisher(Int32, COLLISIONS_TOPIC,
                                                     volatile)
        # Outcome flag (True = all drones reached their targets).
        self._goal_reached_pub = self.create_publisher(Bool,
                                                       GOAL_REACHED_TOPIC,
                                                       volatile)

        self._started = False
        self._go_time_ns = 0
        self._bag_proc = None
        self._bag_dir = None
        self._tlog_mav = None
        self._tlog_fh = None
        self._tlog_subs = []
        self._tlog_ts_ms0 = 0
        self._tlog_ts_us = [0]
        self._tlog_yaw = {prefix: 0.0 for prefix in self.prefixes}
        self._ts = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H-%M-%S')
        self._elapsed = 0.0
        self._view_wait_clock = 0.0
        self._view_wait_latched = False
        self._view_arm_clock = 0.0
        self._view_arm_latched = False
        self._finished = False
        self.timer = self.create_timer(self.interval, self._check)
        self._start_bag()
        self._start_tlog()
        self.get_logger().info(
            f'Start gate waiting for {len(self.prefixes)} drones to report '
            f'ready (timeout {self.timeout:.0f} s, goal timeout '
            f'{self.goal_timeout:.0f} s).')

    def _ready_cb(self, prefix, msg):
        self._ready[prefix] = msg.data

    def _goal_cb(self, prefix, msg):
        if not msg.data or self._finished:
            return
        self._goal[prefix] = True
        if self._started and all_goals(self._goal, self.prefixes):
            self._publish_elapsed()

    def _collision_cb(self, prefix, msg):
        if msg.data:
            self._collided[prefix] = True
        self._publish_collisions()

    def _publish_collisions(self):
        """Publish the aggregate collision count when it changes."""
        total = collision_count(self._collided)
        if total == self._last_collisions:
            return
        self._last_collisions = total
        msg = Int32()
        msg.data = total
        self._collisions_pub.publish(msg)

    def _publish_elapsed(self):
        now_ns = self.get_clock().now().nanoseconds
        secs = elapsed_seconds(self._go_time_ns, now_ns)
        msg = Float64()
        msg.data = secs
        self._elapsed_pub.publish(msg)
        self._goal_reached_pub.publish(Bool(data=True))
        self._publish_collisions()
        self._finished = True
        self._stop_tlog()
        self._stop_bag()
        self.get_logger().info(
            f'Maneuver elapsed: {secs:.2f} s '
            f'(all drones reached their targets), '
            f'collisions: {self._last_collisions}')

    def _check(self):
        if self._started:
            if not self._finished:
                now_ns = self.get_clock().now().nanoseconds
                secs = elapsed_seconds(self._go_time_ns, now_ns)
                missing = [p for p in self.prefixes
                           if not self._goal.get(p, False)]
                if all_goals(self._goal, self.prefixes):
                    self._publish_elapsed()
                elif secs >= self.goal_timeout:
                    self.get_logger().error(
                        f'Goal timeout after {secs:.1f} s with '
                        f'{len(missing)} drone(s) still not at target '
                        f'({", ".join(missing)}) - closing sim, '
                        f'publishing timeout value ({self.goal_timeout} s).')
                    msg = Float64()
                    msg.data = self.goal_timeout
                    self._elapsed_pub.publish(msg)
                    self._goal_reached_pub.publish(Bool(data=False))
                    self._publish_collisions()
                    self._finished = True
                    self._stop_tlog()
                    self._stop_bag()
                    self._shutdown_sim()
            return

        self._elapsed += self.interval

        missing = [p for p in self.prefixes if not self._ready.get(p, False)]
        if should_start(self._ready, self.prefixes, self._elapsed, self.timeout):
            missing = [p for p in self.prefixes
                       if not self._ready.get(p, False)]
            if not self._view_wait_latched:
                self._view_wait_clock = self._elapsed
                self._view_wait_latched = True
            if not self._view_ready \
                    and self._elapsed - self._view_wait_clock \
                    < self.view_ready_timeout:
                if missing:
                    self.get_logger().warn(
                        f'Ready timeout after {self._elapsed:.1f} s with '
                        f'{len(missing)} drone(s) still not ready '
                        f'({", ".join(missing)}) - starting anyway.')
                self.get_logger().info(
                    f'Waiting for GUI view_ready (or '
                    f'{self.view_ready_timeout:.1f} s timeout).')
                return
            if missing:
                self.get_logger().warn(
                    f'Ready timeout after {self._elapsed:.1f} s with '
                    f'{len(missing)} drone(s) still not ready '
                    f'({", ".join(missing)}) - starting anyway.')
            else:
                self.get_logger().info(
                    f'All {len(self.prefixes)} drones ready - starting swarm.')
            self._go_time_ns = self.get_clock().now().nanoseconds
            msg = Bool()
            msg.data = True
            self._go_pub.publish(msg)
            self._started = True
        else:
            self.get_logger().debug(
                f'{len(self.prefixes) - len(missing)}/{len(self.prefixes)} '
                'drones ready', throttle_duration_sec=2.0)

    def _start_bag(self):
        """Launch a ``ros2 bag record`` for this run's per-drone telemetry.

        Records each prefix's ``/{prefix}/gps`` and ``/{prefix}/cmd_vel``
        plus the gate's opener/elapsed topics, landed in a fresh
        ``~/.gz/sim/log/<timestamp>/gps_run_<timestamp>`` bag so every run has
        a uniquely named, self-describing recording.  Best-effort: if
        ``ros2 bag`` is missing or fails to start the run continues unrecorded.
        """
        if not self.recording_enabled:
            return
        try:
            ts = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H-%M-%S')
            root = os.path.join(os.path.expanduser('~'), '.gz', 'sim', 'log')
            run_dir = os.path.join(root, ts)
            bag = os.path.join(run_dir, f'gps_run_{ts}')
            topics = [f'/{p}/gps' for p in self.prefixes]
            topics += [f'/{p}/cmd_vel' for p in self.prefixes]
            topics += [GO_TOPIC, ELAPSED_TOPIC]
            self._bag_proc = subprocess.Popen(
                ['ros2', 'bag', 'record', '-o', bag, '--storage', 'sqlite3']
                + topics,
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            time.sleep(0.6)
            if self._bag_proc.poll() is not None:
                err = self._bag_proc.stderr.read().decode('utf-8', 'replace')
                self._bag_proc = None
                self.get_logger().warn(
                    f'ros2 bag died at start: {err.strip()[:200]}')
                return
            os.makedirs(run_dir, exist_ok=True)
            self.get_logger().info(
                f'Recording bag -> {bag} '
                f'({len(self.prefixes) * 2 + 2} topics).')
        except Exception:  # noqa: BLE001
            self._bag_proc = None
            self.get_logger().warn(
                'ros2 bag unavailable - running unrecorded.')

    def _stop_bag(self):
        """Stop the recorder (idempotent) so its sqlite bag flushes cleanly."""
        proc = self._bag_proc
        self._bag_proc = None
        if proc is None:
            return
        try:
            proc.send_signal(signal.SIGINT)
            proc.wait(timeout=10.0)
        except Exception:  # noqa: BLE001
            try:
                proc.kill()
            except Exception:  # noqa: BLE001
                pass
        self.get_logger().info('Bag recording finished.')

    def _stop_tlog(self):
        """Stop the in-process tlog writer (idempotent) and flush its file.

        Closes the file so every buffered MAVLink message reaches disk, then
        clears ``_tlog_fh``/``_tlog_mav`` so late callbacks are no-ops.  Called
        from the maneuver finish, goal-timeout and SIGTERM paths; must run
        before ``_stop_bag`` so both recorders stop cleanly.
        """
        fh = self._tlog_fh
        self._tlog_fh = None
        self._tlog_mav = None
        if fh is None:
            return
        try:
            fh.flush()
            fh.close()
        except Exception:  # noqa: BLE001
            pass
        self.get_logger().info('Tlog recording finished.')

    def _start_tlog(self):
        """Write this run's GPS/cmd_vel to a MAVLink ``.tlog`` in-process.

        plot.ardupilot.org ingests MAVLink tlogs, not rosbags, so this is the
        recorder that actually serves that goal. Subscribes each prefix's
        ``NavSatFix``/``cmd_vel`` in-process (no subprocess, no QoS guessing)
        and serialises NavSatFix -> ``GPS_RAW_INT`` + cmd_vel ->
        ``LOCAL_POSITION_NED`` into a fresh ``gps_run_<ts>.tlog`` next to the
        bag.  Best-effort: a missing pymavlink just skips the tlog.
        """
        if not self.recording_enabled:
            return
        try:
            from pymavlink.dialects.v20 import ardupilotmega as mav
            run_dir = self._bag_dir or os.path.join(
                os.path.expanduser('~'), '.gz', 'sim', 'log', self._ts)
            os.makedirs(run_dir, exist_ok=True)
            fh = open(os.path.join(run_dir, f'gps_run_{self._ts}.tlog'),
                      'wb')
        except Exception:  # noqa: BLE001
            self._tlog_fh = None
            self.get_logger().warn('pymavlink unavailable - no tlog this run.')
            return
        self._tlog_fh = fh
        try:
            self._tlog_mav = mav.MAVLink(
                _TlogFrameWriter(fh, self._tlog_ts_us),
                srcSystem=1, srcComponent=1)
            self._tlog = self._tlog_mav
            self._tlog_ts_ms0 = int(
                self.get_clock().now().nanoseconds / 1_000_000)
            # One HEARTBEAT so plot.ardupilot.org recognises the autopilot:
            # quadcopter vehicle type, ArduPilot autopilot, armed GUIDED mode.
            self._tlog_mav.heartbeat_send(
                mav.MAV_TYPE_QUADROTOR,
                mav.MAV_AUTOPILOT_ARDUPILOTMEGA,
                mav.MAV_MODE_FLAG_SAFETY_ARMED
                | mav.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                4,  # custom_mode GUIDED
                mav.MAV_STATE_ACTIVE)
            self._tlog_fh.flush()
            for prefix in self.prefixes:
                self._tlog_subs.append(self.create_subscription(
                    NavSatFix, f'/{prefix}/gps',
                    lambda msg, p=prefix: self._tlog_gps(p, msg), 10))
                self._tlog_subs.append(self.create_subscription(
                    Twist, f'/{prefix}/cmd_vel',
                    lambda msg, p=prefix: self._tlog_vel(p, msg), 10))
            self.get_logger().info(
                f'Writing tlog -> {self._tlog_fh.name} '
                f'({len(self.prefixes)} drones).')
        except Exception:  # noqa: BLE001
            fh.close()
            self._tlog_fh = None
            self._tlog_mav = None
            self.get_logger().warn('tlog setup failed - no tlog this run.')

    def _tlog_gps(self, prefix, msg):
        """NavSatFix -> GPS_RAW_INT/GLOBAL_POSITION_INT for plot.ardupilot.org.

        ``GLOBAL_POSITION_INT`` is the extractor's first-preferred trajectory
        source, ``GPS_RAW_INT`` the accepted fallback, so both are written.
        """
        if self._tlog_mav is None or self._started and self._finished:
            return
        now_ms = int(self.get_clock().now().nanoseconds / 1_000_000)
        run_ms = max(0, now_ms - self._tlog_ts_ms0)
        fix = 3 if msg.status.status >= 0 else 0
        lat = int(msg.latitude * 1e7)
        lon = int(msg.longitude * 1e7)
        alt = int(msg.altitude * 1000.0)
        self._tlog_ts_us[0] = run_ms * 1000
        self._tlog_mav.gps_raw_int_send(
            run_ms * 1000, fix, lat, lon, alt, 1000, 1000, 0, 0, 0)
        self._tlog_mav.global_position_int_send(
            run_ms, lat, lon, alt, alt, 0, 0, 0, 0)
        # ATTITUDE is plot.ardupilot.org's hard map gate (the Cesium view
        # refuses to render without it), so emit it alongside every fix using
        # the drone's latest command heading from cmd_vel as the yaw.
        self._tlog_mav.attitude_send(
            run_ms, 0.0, 0.0, self._tlog_yaw.get(prefix, 0.0),
            0.0, 0.0, 0.0)
        self._tlog_fh.flush()

    def _tlog_vel(self, prefix, msg):
        """cmd_vel -> LOCAL_POSITION_NED so the plot shows a 3-D path."""
        if self._tlog_mav is None or self._started and self._finished:
            return
        now_ms = int(self.get_clock().now().nanoseconds / 1_000_000)
        run_ms = max(0, now_ms - self._tlog_ts_ms0)
        self._tlog_yaw[prefix] = math.atan2(msg.linear.y, msg.linear.x)
        self._tlog_ts_us[0] = run_ms * 1000
        self._tlog_mav.local_position_ned_send(
            run_ms, 0.0, 0.0, 0.0,
            msg.linear.x, msg.linear.y, msg.linear.z, 0.0)
        self._tlog_fh.flush()

    def _shutdown_sim(self):
        """Terminate the gz-sim session this gate belongs to.

        gz-sim 8 has no ROS shutdown service/topic, so this best-effort kills
        the session's ``gz sim`` server.  The launch's ``OnProcessExit`` handler
        then tears the rest of the stack down.
        """
        try:
            for pat in ('gz-sim-server', 'gz sim'):
                subprocess.Popen(
                    ['pkill', '-f', pat],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
        except Exception:  # noqa: BLE001
            pass


def main(args=None):
    rclpy.init(args=args)
    node = DroneStartGateNode()

    def _on_term(signum, _frame):
        """Stop recorders and exit cleanly if the gate is killed.

        The launch tree already SIGTERMs this node on shutdown, but when the
        launch parent itself is SIGKILLed the children are reparented and the
        gate's ``ros2 bag record`` subprocess would otherwise survive forever
        as an orphan.  Trap the signal here to tear tlog/bag down ourselves.
        """
        node.get_logger().info(
            f'Signal {signum} - stopping recorders and exiting.')
        try:
            node._stop_tlog()
            node._stop_bag()
        finally:
            raise SystemExit(0)

    signal.signal(signal.SIGTERM, _on_term)
    signal.signal(signal.SIGINT, _on_term)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except SystemExit:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()