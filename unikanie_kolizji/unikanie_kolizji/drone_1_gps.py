#!/usr/bin/env python3

import math

import gz.transport13 as gz
from gz.msgs10 import pose_v_pb2
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import Bool


# Gazebo world the drones are simulated in. The real model poses are read back
# from /world/<world>/dynamic_pose/info so navigation and avoidance act on the
# actual simulated bodies instead of a self-integrated dead-reckoning estimate.
WORLD_NAME = 'empty_world_harmonic'
POSE_TOPIC = f'/world/{WORLD_NAME}/dynamic_pose/info'


class RealPoseSubscriber:
    """Latest real world pose (x, y, z) of a model, read back from Gazebo.

    Subscribes to the simulation's dynamic_pose/info feed and keeps the newest
    pose of the requested model. This closes the position loop: the controller
    and the collision avoidance react to where the drone really is.
    """

    def __init__(self, model_name):
        self.model_name = model_name
        self.pos = None
        self._node = gz.Node()
        self._node.subscribe(pose_v_pb2.Pose_V, POSE_TOPIC, self._cb)

    def _cb(self, msg):
        for pose in msg.pose:
            if pose.name == self.model_name:
                self.pos = (pose.position.x, pose.position.y, pose.position.z)
                return

    def get(self):
        return self.pos


class Drone1GPSNode(Node):
    """Drone 1 controller (real dynamic flight, closed-loop position).

    Drives the 'drone_left' model toward its target using the gz-sim
    VelocityControl system. A base controller computes a world-frame linear
    velocity pointing toward the target from the drone's REAL pose (read back
    from /world/<world>/dynamic_pose/info); the collision-avoidance command
    (published on /drone_1/cmd_vel by the avoidance node) is summed in as an
    offset. The fused command is published on /drone_1/drive_cmd, which
    ros_gz_bridge forwards to /model/drone_left/cmd_vel inside Gazebo.
    """

    DT = 0.1

    def __init__(self):
        super().__init__('drone_1_gps')
        self.model_name = 'drone_left'
        self.speed = 0.4
        self.stop_radius = 0.25
        self.max_xy_speed = 1.0
        self.max_z_speed = 1.0

        self.method = self.declare_parameter(
            'collision_avoidance_method', 'method_1').get_parameter_value().string_value

        self.start_world = (-2.0, 0.0, 0.5)
        self.target_world = (2.0, 5.0, 1.0)

        self.pose_source = RealPoseSubscriber(self.model_name)

        self.drive_pub = self.create_publisher(Twist, '/drone_1/drive_cmd', 10)
        self.gps_pub = self.create_publisher(NavSatFix, '/drone_1/gps', 10)
        self.cmd_vel_sub = self.create_subscription(
            Twist, '/drone_1/cmd_vel', self.cmd_vel_cb, 10)

        self.enable_pub = self.create_publisher(Bool, '/drone_left/enable', 10)
        self.enable_timer = self.create_timer(0.5, self._publish_enable)
        self._enable_count = 0

        self.avoidance_vel = None
        self.avoidance_scale = 1.0

        self.timer = self.create_timer(self.DT, self.update)
        self.get_logger().info(
            'Drone 1 velocity controller started: flying to (2, 5, 1), '
            'listening for /drone_1/cmd_vel avoidance commands.')

    def _publish_enable(self):
        msg = Bool()
        msg.data = True
        self.enable_pub.publish(msg)
        self._enable_count += 1
        if self._enable_count >= 10:
            self.enable_timer.cancel()

    def cmd_vel_cb(self, msg):
        self.avoidance_vel = msg

    def world_to_gps(self, x, y, z=0.0):
        lat = 50.0 + x / 111320.0
        lon = 19.0 + y / (111320.0 * math.cos(math.radians(50.0)))
        return lat, lon, z

    def update(self):
        real = self.pose_source.get()
        if real is None:
            self.get_logger().debug(
                'Waiting for first real pose from Gazebo...', throttle_duration_sec=2.0)
            return
        cx, cy, cz = real

        if self.method in ('method_2', 'method_3', 'method_4',
                           'method_5', 'method_6'):
            # Full-reference methods (3D-SWAP, DPSO, convex hull, GMM,
            # combined) fully determine the reference velocity (horizontal and
            # vertical); forward it as-is, only bounding the magnitude so the
            # commanded speeds stay within what the physics can honour.
            drive = Twist()
            if self.avoidance_vel is not None:
                drive.linear.x = self.avoidance_vel.linear.x
                drive.linear.y = self.avoidance_vel.linear.y
                drive.linear.z = self.avoidance_vel.linear.z
                xy_norm = math.hypot(drive.linear.x, drive.linear.y)
                if xy_norm > self.max_xy_speed:
                    k = self.max_xy_speed / xy_norm
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

            drive = Twist()
            if dist > self.stop_radius:
                vx = dx / dist * self.speed
                vy = dy / dist * self.speed
            else:
                vx = 0.0
                vy = 0.0
            drive.linear.x = float(vx)
            drive.linear.y = float(vy)
            drive.linear.z = float(max(dz * 0.5, 0.0))

            # Blend in the collision-avoidance offset.
            if self.avoidance_vel is not None:
                drive.linear.x += self.avoidance_vel.linear.x * self.avoidance_scale
                drive.linear.y += self.avoidance_vel.linear.y * self.avoidance_scale

            # Bound the fused XY command so avoidance cannot fling the drone away
            # faster than it can recover toward the target.
            xy_norm = math.hypot(drive.linear.x, drive.linear.y)
            if xy_norm > self.max_xy_speed:
                k = self.max_xy_speed / xy_norm
                drive.linear.x *= k
                drive.linear.y *= k

        self.drive_pub.publish(drive)

        lat, lon, alt = self.world_to_gps(cx, cy, cz)
        msg = NavSatFix()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.latitude = lat
        msg.longitude = lon
        msg.altitude = alt
        self.gps_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = Drone1GPSNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
