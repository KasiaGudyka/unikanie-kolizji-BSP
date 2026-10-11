#!/usr/bin/env python3

import math

import gz.transport13 as gz
from gz.msgs10 import pose_v_pb2
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import Bool

from unikanie_kolizji.world_generator import WORLD_NAME


# Gazebo world the drones are simulated in. The real model poses are read back
# from /world/<world>/dynamic_pose/info so navigation and avoidance act on the
# actual simulated bodies instead of a self-integrated dead-reckoning estimate.
POSE_TOPIC = f'/world/{WORLD_NAME}/dynamic_pose/info'

# Topic used by the start gate node to release every drone into the maneuver at
# the same time, once all drones have confirmed they sit at their start poses.
# View-ready handshake topic: the GUI latches a Bool(True) here once its 3-D
# panel has rendered a run, so drones sit parked (ready) until the maneuver is
# actually visible.  Latched TRANSIENT_LOCAL on both sides so the start gate
# reliably receives the GUI's signal even when the gate latches the topic
# before the GUI publishes.
VIEW_READY_TOPIC = '/simulation/view_ready'

GO_TOPIC = '/simulation/go'

# Topic on which the start gate publishes the measured maneuver time (seconds,
# std_msgs/Float64) once every drone has reported goal_reached (or the run times
# out).  Volatile, not latched: a previous run's value must not stick to a fresh
# run in the GUI.
ELAPSED_TOPIC = '/simulation/elapsed'

# Collision-cylinder semantics for proximity detection, matching 3D-SWAP's
# "occupied space" cylinder (rc = 0.35). A drone declares itself in collision
# when another drone's centre is within COLLISION_RADIUS_M horizontally AND
# COLLISION_VERTICAL_M vertically of its own. The start gate latches every
# drone's /<prefix>/collision Bool and divides the involved drones by two.
COLLISION_RADIUS_M = 0.7
COLLISION_VERTICAL_M = 0.6

# GPS reference frame shared with the avoidance node and the trajectory panel
# (world xy = (lat - REF_LAT) * 111320, y scaled by cos(REF_LAT)).
REF_LAT = 50.0
REF_LON = 19.0

# "Gentle heavy" guidance profile: heavier drones fly sedately like a commercial
# cargo UAV (capped acceleration, deceleration ramp into the goal, slightly
# slower cruise). The 0.5 kg entry matches the legacy cruise speed but now also
# rate-limits the reference velocity (max_accel) so raw setpoint steps from the
# base controller or the avoidance planners cannot excite the inner attitude
# loop into a tumble; the small quad previously bypassed the ramp entirely.
# wind_cap_xy / wind_k give a wind-aware XY authority:
# while the drone is near its waypoint the nominal max_xy_speed applies, but
# once a steady wind blows it off track the reference ramps with the drift so a
# real-airframe quad can hold position (or claw back) instead of sailing away.
# Wind flight is anchored by two more mechanisms: wind worlds spawn armed at
# cruise height and gate readiness on stable altitude (so the 0.25 m start gate
# cannot be blown open before 'go'), and every command is rotated into the
# vehicle body frame (MulticopterVelocityControl feeds its velocity error with
# R * cmd). Values are tuned / validated live in the headless sim.
GUIDANCE_PROFILE = {
    0.5: {'speed': 0.4, 'max_xy_speed': 1.0, 'max_z_speed': 1.0,
          'brake_dist': 0.0, 'max_accel': 2.0, 'z_gain': 0.5,
          'wind_cap_xy': 4.0, 'wind_k': 0.5},
    5.0: {'speed': 0.35, 'max_xy_speed': 0.9, 'max_z_speed': 0.5,
          'brake_dist': 2.0, 'max_accel': 1.0, 'z_gain': 0.22,
          'wind_cap_xy': 6.0, 'wind_k': 0.5},
    20.0: {'speed': 0.35, 'max_xy_speed': 0.75, 'max_z_speed': 0.4,
           'brake_dist': 3.0, 'max_accel': 0.9, 'z_gain': 0.18,
           'wind_cap_xy': 5.0, 'wind_k': 0.5},
}


def near_start(pos, start_world, tolerance):
    """True when a real pose is within ``tolerance`` metres of the start."""
    if pos is None or start_world is None:
        return False
    d = math.sqrt(sum((a - b) ** 2
                      for a, b in zip(pos[:3], start_world[:3])))
    return d <= tolerance


def in_goal_zone(pos, goal_world, radius):
    """True when a real pose is within ``radius`` metres of the goal."""
    return near_start(pos, goal_world, radius)


def gps_to_xy(lat, lon):
    """Convert a NavSatFix lat/lon back into world x/y metres.

    Mirrors the avoidance node / trajectory panel conversion so neighbour
    positions read from ``/<prefix>/gps`` land in the same world frame as the
    real poses read from Gazebo.
    """
    x = (lat - REF_LAT) * 111320.0
    y = (lon - REF_LON) * 111320.0 * math.cos(math.radians(REF_LAT))
    return x, y


def drones_in_contact(own, neighbors, radius=COLLISION_RADIUS_M,
                      vertical=COLLISION_VERTICAL_M):
    """True when any neighbour centre is inside the collision cylinder.

    ``own`` / ``neighbors`` are (x, y, z) world metres. A contact is declared
    when a neighbour is closer than ``radius`` horizontally and ``vertical``
    vertically - the same occupied-space semantics 3D-SWAP uses for its
    collision cylinder.
    """
    cx, cy, cz = own
    for nx, ny, nz in neighbors:
        if math.hypot(nx - cx, ny - cy) < radius and abs(nz - cz) < vertical:
            return True
    return False


class RealPoseSubscriber:
    """Latest real world pose (x, y, z) of a model, read back from Gazebo.

    Subscribes to the simulation's dynamic_pose/info feed and keeps the newest
    pose of the requested model. This closes the position loop: the controller
    and the collision avoidance react to where the drone really is.
    """

    def __init__(self, model_name):
        self.model_name = model_name
        self.pos = None
        self.heading = None
        self.attitude = None
        self._node = gz.Node()
        self._node.subscribe(pose_v_pb2.Pose_V, POSE_TOPIC, self._cb)

    def _cb(self, msg):
        body = None
        for pose in msg.pose:
            if pose.name == self.model_name:
                self.pos = (pose.position.x, pose.position.y, pose.position.z)
            elif pose.name == 'body':
                # gz publishes link poses with non-namespaced names, so the
                # 'body' entry appears once per drone. Match by proximity to
                # the tracked model's own position.
                if self.pos is not None:
                    dx = pose.position.x - self.pos[0]
                    dy = pose.position.y - self.pos[1]
                    if dx * dx + dy * dy < 4.0:
                        body = pose.orientation
                elif body is None:
                    body = pose.orientation
        if body is not None:
            self._from_quat(body)

    def _from_quat(self, q):
        self.heading = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        # Tait-Bryan roll/pitch (x-then-y order) from the body
        # quaternion, for attitude-instability telemetry.
        roll = math.atan2(
            2.0 * (q.w * q.x + q.y * q.z),
            1.0 - 2.0 * (q.x * q.x + q.y * q.y))
        pitch = math.asin(
            max(-1.0, min(1.0, 2.0 * (q.w * q.y - q.z * q.x))))
        self.attitude = (roll, pitch)

    def get(self):
        return self.pos


class DroneGPSNode(Node):
    """One parametric drone controller (real dynamic flight, closed-loop pose).

    Drives the Gazebo model named by ``model_name`` toward its own
    ``target_world`` waypoint using the gz-sim MulticopterVelocityControl
    system. The launch file instantiates one of these per drone with unique
    parameters, replacing the former hard-coded drone_1/drone_2 nodes.

    A base controller computes a world-frame linear velocity pointing toward
    the target from the drone's REAL pose (read back from
    /world/<world>/dynamic_pose/info); the collision-avoidance command
    (published on /<topic_prefix>/cmd_vel by the avoidance node) is summed in
    as an offset. The fused command is converted to the vehicle body frame
    (since MulticopterVelocityControl interprets cmd_vel in the body frame)
    and published on /<topic_prefix>/drive_cmd, which ros_gz_bridge forwards
    to /<model_name>/cmd_vel inside Gazebo.
    """

    DT = 0.1

    def __init__(self):
        super().__init__('drone_gps')
        self.model_name = self.declare_parameter(
            'model_name', 'drone_0').get_parameter_value().string_value
        self.topic_prefix = self.declare_parameter(
            'topic_prefix', self.model_name).get_parameter_value().string_value
        self.target_world = tuple(
            self.declare_parameter(
                'target_world', [2.0, 5.0, 1.0]).get_parameter_value()
            .double_array_value)
        self.start_world = tuple(
            self.declare_parameter(
                'start_world', [0.0, 0.0, 0.5]).get_parameter_value()
            .double_array_value)
        self.ready_tolerance = self.declare_parameter(
            'ready_tolerance', 0.25).get_parameter_value().double_value
        self.ready_settle_ticks = self.declare_parameter(
            'ready_settle_ticks', 4).get_parameter_value().integer_value
        self.method = self.declare_parameter(
            'collision_avoidance_method', 'method_1').get_parameter_value().string_value
        self.neighbor_prefixes = list(self.declare_parameter(
            'neighbor_prefixes', Parameter.Type.STRING_ARRAY).get_parameter_value().string_array_value)
        self.collision_radius = self.declare_parameter(
            'collision_radius', COLLISION_RADIUS_M).get_parameter_value().double_value
        self.collision_vertical = self.declare_parameter(
            'collision_vertical', COLLISION_VERTICAL_M).get_parameter_value().double_value
        self.drone_mass_kg = self.declare_parameter(
            'drone_mass_kg', 0.5).get_parameter_value().double_value
        self.wind = self.declare_parameter(
            'wind', 'off').get_parameter_value().string_value
        self.wind_mode = self.wind != 'off'
        profile = GUIDANCE_PROFILE.get(
            self.drone_mass_kg, GUIDANCE_PROFILE[0.5])
        self.speed = self.declare_parameter(
            'speed', profile['speed']).get_parameter_value().double_value
        self.stop_radius = self.declare_parameter(
            'stop_radius', 0.25).get_parameter_value().double_value
        self.goal_settle_ticks = self.declare_parameter(
            'goal_settle_ticks', 3).get_parameter_value().integer_value
        self.max_xy_speed = self.declare_parameter(
            'max_xy_speed', profile['max_xy_speed']).get_parameter_value().double_value
        self.max_z_speed = self.declare_parameter(
            'max_z_speed', profile['max_z_speed']).get_parameter_value().double_value
        self.brake_dist = profile['brake_dist']
        self.max_accel = profile['max_accel']
        self.z_gain = profile['z_gain']
        self.wind_cap_xy = profile['wind_cap_xy']
        self.wind_k = profile['wind_k']

        # Gentle-heavy velocity-rate limiter state (unused for 0.5 kg).
        # Gentle-heavy velocity-rate limiter state (unused for 0.5 kg).
        self._prev_vel = (0.0, 0.0, 0.0)
        self._prev_pos = None
        self._retreat_ticks = 0
        self._recovery = False
        self._xy_cap = self.max_xy_speed
        self._snapped = False

        self.pose_source = RealPoseSubscriber(self.model_name)

        drive_topic = f'/{self.topic_prefix}/drive_cmd'
        gps_topic = f'/{self.topic_prefix}/gps'
        cmd_vel_topic = f'/{self.topic_prefix}/cmd_vel'
        enable_topic = f'/{self.model_name}/enable'
        ready_topic = f'/{self.topic_prefix}/ready'
        goal_topic = f'/{self.topic_prefix}/goal_reached'
        collision_topic = f'/{self.topic_prefix}/collision'

        self.drive_pub = self.create_publisher(Twist, drive_topic, 10)
        self.gps_pub = self.create_publisher(NavSatFix, gps_topic, 10)
        self.cmd_vel_sub = self.create_subscription(
            Twist, cmd_vel_topic, self.cmd_vel_cb, 10)

        transient = QoSProfile(depth=10,
                               durability=DurabilityPolicy.TRANSIENT_LOCAL)

        # Proximity-collision reporting. Subscribe to every neighbour's GPS
        # feed (same topics the avoidance node consumes) and publish a Bool on
        # /<prefix>/collision while another drone is inside the collision
        # cylinder during the maneuver. The start gate latches + halves these.
        self._neighbor_pos = {}
        self.neighbor_subs = []
        for pref in self.neighbor_prefixes:
            if pref == self.topic_prefix:
                continue
            self.neighbor_subs.append(self.create_subscription(
                NavSatFix, f'/{pref}/gps',
                lambda msg, p=pref: self.neighbor_gps_cb(p, msg), 10))
        self.collision_pub = self.create_publisher(Bool, collision_topic,
                                                   transient)

        self.ready_pub = self.create_publisher(Bool, ready_topic, transient)
        self.goal_reached_pub = self.create_publisher(Bool, goal_topic,
                                                      transient)
        self.go_sub = self.create_subscription(
            Bool, GO_TOPIC, self.go_cb, transient)

        self.enable_pub = self.create_publisher(Bool, enable_topic, 10)
        self.enable_timer = self.create_timer(0.5, self._publish_enable)

        self.avoidance_vel = None
        self.avoidance_scale = 1.0

        # Readiness handshake state: the drone holds at its start pose and is
        # only released into the maneuver (drive + gps) once it has been stable
        # near `start_world` for `ready_settle_ticks` ticks AND the start gate
        # published a global `go`.
        self.ready = False
        self._ready_ticks = 0
        self._go = False

        # Goal-reached reporting: publishes on /<prefix>/goal_reached exactly
        # once, after the drone has been in the goal zone for a few ticks.
        self._goal_reported = False
        self._goal_ticks = 0

        self.timer = self.create_timer(self.DT, self.update)
        self.get_logger().info(
            f'Drone controller for {self.model_name} started: '
            f'flying to {tuple(round(v, 2) for v in self.target_world)}, '
            f'start {tuple(round(v, 2) for v in self.start_world)}, '
            'listening for avoidance commands on '
            f'{cmd_vel_topic}.')

    def _publish_enable(self):
        msg = Bool()
        msg.data = True
        self.enable_pub.publish(msg)

    def cmd_vel_cb(self, msg):
        self.avoidance_vel = msg

    def go_cb(self, msg):
        if msg.data:
            self._go = True

    def neighbor_gps_cb(self, prefix, msg):
        """Track a neighbour drone's world position from its GPS feed."""
        x, y = gps_to_xy(msg.latitude, msg.longitude)
        self._neighbor_pos[prefix] = (x, y, msg.altitude)

    def _update_collision(self, cx, cy, cz):
        """Publish the proximity-collision state during the maneuver.

        A drone reports True on /<prefix>/collision while any neighbour centre
        is inside its collision cylinder. The start gate latches the report and
        counts distinct drones (each physical collision involves exactly two,
        so the gate halves the total). Reporting is stateful per tick so the
        count never rises with the contact duration.
        """
        msg = Bool()
        msg.data = bool(drones_in_contact(
            (cx, cy, cz), self._neighbor_pos.values(),
            self.collision_radius, self.collision_vertical))
        self.collision_pub.publish(msg)

    def world_to_gps(self, x, y, z=0.0):
        lat = 50.0 + x / 111320.0
        lon = 19.0 + y / (111320.0 * math.cos(math.radians(50.0)))
        return lat, lon, z

    def _hold_twist(self, cx, cy, cz):
        """Twist that steers the drone back to and holds its start pose."""
        tx, ty, tz = self.start_world
        dx = tx - cx
        dy = ty - cy
        dz = tz - cz
        dist = math.hypot(dx, dy)

        drive = Twist()
        if dist > self.stop_radius:
            vx = dx / dist * self.speed
            vy = dy / dist * self.speed
        else:
            vx = 0.0
            vy = 0.0
        drive.linear.x = float(vx)
        drive.linear.y = float(vy)
        drive.linear.z = float(self._z_cmd(dz))
        return drive

    def _hover_hold(self, cz, start_z):
        """Wind-world pre-go hover: hold altitude, drift freely in x/y.

        The gz velocity controller has no position hold against a constant
        wind, so trying to hold the start x/y would just pitch the drone into a
        losing fight. Command zero horizontal velocity and steady the altitude
        instead; the recovery loop takes over once 'go' arrives.
        """
        drive = Twist()
        drive.linear.z = float(self._z_cmd(start_z - cz))
        return drive

    def _publish_ready(self):
        msg = Bool()
        msg.data = True
        self.ready_pub.publish(msg)

    def _z_cmd(self, dz):
        """Vertical velocity toward the reference altitude.

        Legacy 0.5 kg profile stays up-only (never commanded below the start
        altitude); heavy profiles command a signed, gentle descent so an
        over-height drone settles back to its reference instead of free-hanging
        (which broke the pre-flight readiness handshake).
        """
        if self.z_gain >= 0.5:
            return max(dz * 0.5, 0.0)
        return dz * self.z_gain

    def _effective_xy_cap(self) -> float:
        """Wind-aware XY speed bound (see update())."""
        return getattr(self, '_xy_cap', self.max_xy_speed)

    def _ramp_velocity(self, drive):
        """Limit the commanded velocity change per tick (gentle acceleration).

        Applied to every mass profile (0.5 kg with max_accel=2.0 included):
        a raw setpoint step from the base controller's goal-line reversal or
        stop-radius snap, or from a full-reference planner's replan, would
        otherwise demand an attitude spike the small body cannot damp.
        Vertical is gentled to half the horizontal rate so the drone does not
        carry through its target altitude.
        """
        if self.max_accel <= 0.0:
            return
        lim = self.max_accel * self.DT
        lim_z = lim * 0.5
        cur = (drive.linear.x, drive.linear.y, drive.linear.z)
        blended = (
            fr + math.copysign(lim, to - fr) if abs(to - fr) > lim else to
            for fr, to in zip(self._prev_vel, cur))
        prev_x, prev_y, prev_z = self._prev_vel
        target_z = cur[2]
        if abs(target_z - prev_z) > lim_z:
            blended_z = prev_z + math.copysign(lim_z, target_z - prev_z)
        else:
            blended_z = target_z
        blended = tuple(blended)
        drive.linear.x, drive.linear.y = blended[0], blended[1]
        drive.linear.z = blended_z
        self._prev_vel = (blended[0], blended[1], blended_z)

    def _to_body_frame(self, vx, vy):
        """Rotate a world-frame velocity into the vehicle body frame.

        Gazebo's MulticopterVelocityControl interprets ``cmd_vel`` in the
        vehicle body frame (its velocity error is ``v_world - R * cmd``), so a
        world-frame reference only tracks when the heading is zero. Rotating by
        the drone's current yaw keeps the wind fight and the calm cruise on the
        true world trajectory. No pose yet -> identity.
        """
        heading = self.pose_source.heading
        if heading is None:
            return vx, vy
        c, s = math.cos(heading), math.sin(heading)
        return vx * c + vy * s, -vx * s + vy * c

    def update(self):
        real = self.pose_source.get()
        if real is None:
            self.get_logger().debug(
                'Waiting for first real pose from Gazebo...', throttle_duration_sec=2.0)
            return
        cx, cy, cz = real

        # Wind-aware XY authority: nominal sedate max_xy_speed rules unless the
        # wind is actually pushing the airframe away from its waypoint despite
        # us commanding toward it (retreat detection). Only then does the
        # toward-target reference ramp with the drift (up to wind_cap_xy) so a
        # real-airframe quad claws back instead of sailing away. Detection is
        # gated on wind_mode: in calm runs an avoidance detour momentarily
        # steers a drone off its goal line, which would otherwise be mistaken
        # for a wind retreat and balloon the cap to wind_cap_xy (3-4 m/s) on a
        # 0.5 kg body -- the root cause of the unrouted "spontaneous" flips.
        gx, gy = self.target_world[0], self.target_world[1]
        drift = math.hypot(cx - gx, cy - gy)
        vx = vy = 0.0
        if self._prev_pos is not None:
            vx = (cx - self._prev_pos[0]) / self.DT
            vy = (cy - self._prev_pos[1]) / self.DT
        self._prev_pos = (cx, cy)
        if (self.wind_mode and drift > 0.5
                and vx * (gx - cx) + vy * (gy - cy) < -0.05 * drift):
            self._retreat_ticks += 1
        else:
            self._retreat_ticks = max(0, self._retreat_ticks - 1)
        self._recovery = self._retreat_ticks >= 5
        if self._recovery:
            self._xy_cap = min(self.wind_cap_xy,
                               max(self.max_xy_speed, self.wind_k * drift))
        else:
            self._xy_cap = self.max_xy_speed

        # Readiness handshake: hold at the start pose (and stay silent on GPS)
        # until the drone has been stable near `start_world` for a few ticks.
        #
        # In wind worlds the drone spawns airborne at cruise height; the strict
        # 0.25 m start-zone gate would be blown open by the wind in the first
        # second (the drone then never reports ready and just sails while
        # weakly holding toward its start). Wind readiness therefore means
        # "has reached a stable hover altitude" - the start x/y is allowed to
        # ride the wind until the gate sends 'go' and the recovery loop claws
        # it back to the target. Calm worlds keep the legacy strict gate.
        if not self.ready:
            start_z = self.start_world[2]
            if self.wind_mode:
                settled = abs(cz - start_z) < 0.6
            else:
                settled = near_start((cx, cy, cz), self.start_world,
                                     self.ready_tolerance)
            if settled:
                self._ready_ticks += 1
            else:
                self._ready_ticks = 0
            if self._ready_ticks >= self.ready_settle_ticks:
                self.ready = True
                self.get_logger().info(
                    f'{self.model_name} ready at start pose '
                    f'({cx:.3f}, {cy:.3f}, {cz:.3f}) - waiting for go.')
            else:
                if self.wind_mode:
                    self.drive_pub.publish(self._hover_hold(cz, start_z))
                else:
                    self.drive_pub.publish(self._hold_twist(cx, cy, cz))
                return

        if self.ready:
            self._publish_ready()

        # Global start gate: no maneuver / trajectory data until all drones
        # have confirmed their start positions and the gate sent the go signal.
        if not self._go:
            self.drive_pub.publish(self._hold_twist(cx, cy, cz))
            return

        # Goal-reached detection.  The target spheres in the world are visual
        # only, so a drone reaches its target by proximity (no collision).
        # Report once the drone has been inside the goal zone for a few ticks
        # (debounces overshoot/fluke crossings) and latch the report so the
        # gate can still collect it after the drone drifts away.
        gz1, gz2, gz3 = self.target_world
        goal_dist = math.sqrt((cx - gz1) ** 2 + (cy - gz2) ** 2 + (cz - gz3) ** 2)
        if goal_dist <= self.stop_radius:
            self._goal_ticks += 1
        else:
            self._goal_ticks = 0
        if (not self._goal_reported
                and self._goal_ticks >= self.goal_settle_ticks):
            self._goal_reported = True
            goal_msg = Bool()
            goal_msg.data = True
            self.goal_reached_pub.publish(goal_msg)
            self.get_logger().info(
                f'{self.model_name} reached its target at '
                f'({cx:.3f}, {cy:.3f}, {cz:.3f}).')

        if self.method in ('method_2', 'method_3', 'method_4',
                           'method_5', 'method_6'):
            # Full-reference methods (3D-SWAP, DPSO, convex hull, GMM, combined)
            # fully determine the reference velocity (horizontal and
            # vertical); forward it as-is and bound the magnitude with the
            # wind-aware cap. When the airframe is being pushed away by a
            # steady wind (retreat detected), the toward-target recovery
            # reference takes over so the quad claws back instead of sailing
            # away; an actively steering avoidance command is folded in at
            # half weight.
            drive = Twist()
            av = self.avoidance_vel
            if av is not None:
                drive.linear.x = av.linear.x
                drive.linear.y = av.linear.y
                drive.linear.z = av.linear.z
            if self._recovery:
                rec_mag = min(self.wind_cap_xy, self.wind_k * drift)
                drive.linear.x = (gx - cx) / drift * rec_mag
                drive.linear.y = (gy - cy) / drift * rec_mag
                if av is not None and math.hypot(av.linear.x, av.linear.y) > 0.2:
                    drive.linear.x += 0.5 * av.linear.x
                    drive.linear.y += 0.5 * av.linear.y
            xy_norm = math.hypot(drive.linear.x, drive.linear.y)
            xy_cap = self._effective_xy_cap()
            if xy_norm > xy_cap:
                k = xy_cap / xy_norm
                drive.linear.x *= k
                drive.linear.y *= k
            if abs(drive.linear.z) > self.max_z_speed:
                drive.linear.z = math.copysign(self.max_z_speed,
                                                drive.linear.z)
        else:
            tx, ty, tz = self.target_world

            dx = tx - cx
            dy = ty - cy
            dz = tz - cz
            dist = math.hypot(dx, dy)

            self._snapped = (dist <= self.stop_radius)
            drive = Twist()
            if dist > self.stop_radius:
                sp = self.speed
                if self.brake_dist > 0.0:
                    # Gentle-heavy deceleration ramp: bleed speed to zero
                    # while approaching so the drone settles instead of
                    # overshooting the goal.
                    sp *= min(1.0, dist / self.brake_dist)
                if self._recovery:
                    # Pushed away by the wind: full recovery velocity.
                    sp = min(self.wind_cap_xy, self.wind_k * dist)
                vx = dx / dist * sp
                vy = dy / dist * sp
            else:
                vx = 0.0
                vy = 0.0
            drive.linear.x = float(vx)
            drive.linear.y = float(vy)
            drive.linear.z = float(self._z_cmd(dz))

            # Blend in the collision-avoidance offset.
            if self.avoidance_vel is not None:
                drive.linear.x += self.avoidance_vel.linear.x * self.avoidance_scale
                drive.linear.y += self.avoidance_vel.linear.y * self.avoidance_scale

            # Bound the fused XY command so avoidance cannot fling the drone away
            # faster than it can recover toward the target.
            xy_norm = math.hypot(drive.linear.x, drive.linear.y)
            xy_cap = self._effective_xy_cap()
            if xy_norm > xy_cap:
                k = xy_cap / xy_norm
                drive.linear.x *= k
                drive.linear.y *= k

        self._ramp_velocity(drive)
        drive.linear.x, drive.linear.y = self._to_body_frame(
            drive.linear.x, drive.linear.y)

        self.drive_pub.publish(drive)

        att = self.pose_source.attitude
        if att is not None:
            self.get_logger().info(
                f'{self.model_name} att=(r{math.degrees(att[0]):+8.1f} '
                f'p{math.degrees(att[1]):+8.1f} '
                f'y{math.degrees(self.pose_source.heading):+8.1f}) '
                f'cmd=({drive.linear.x:+.3f},{drive.linear.y:+.3f},'
                f'{drive.linear.z:+.3f}) snap={getattr(self, "_snapped", False)}')

        lat, lon, alt = self.world_to_gps(cx, cy, cz)
        msg = NavSatFix()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.latitude = lat
        msg.longitude = lon
        msg.altitude = alt
        self.gps_pub.publish(msg)

        self._update_collision(cx, cy, cz)


def main(args=None):
    rclpy.init(args=args)
    node = DroneGPSNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()