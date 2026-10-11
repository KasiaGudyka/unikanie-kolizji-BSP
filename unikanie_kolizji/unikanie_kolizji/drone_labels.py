#!/usr/bin/env python3

"""Publish Gazebo TEXT markers above each drone (gz-sim-markers-system).

Sits above every spawned drone for the duration of the simulation and re-
publishes a ``gz.msgs.Marker`` (type ``TEXT``) whose text and colour match the
drone name and its assigned target colour.  The ``gz-sim-markers-system``
plugin must be present in the world for these markers to render.

The position is read back from Gazebo's ``dynamic_pose/info`` stream exactly
as ``drone_gps.RealPoseSubscriber`` does, so the labels track the real
simulated bodies.
"""

import math

import gz.transport13 as gz
from gz.msgs10 import marker_pb2
import rclpy
from rclpy.node import Node

from unikanie_kolizji.drone_gps import RealPoseSubscriber
from unikanie_kolizji.world_generator import color_rgb, WORLD_NAME


def build_label_marker(name: str, drone_index: int, x: float, y: float,
                       z: float, offset_z: float = 0.55,
                       scale: float = 0.35) -> marker_pb2.Marker:
    """Return a persistent TEXT marker for one drone at (x, y, z)."""
    r, g, b = color_rgb(drone_index)
    m = marker_pb2.Marker()
    m.ns = 'unikanie_kolizji_labels'
    m.id = drone_index + 1          # unique per drone
    m.type = marker_pb2.Marker.TEXT
    m.action = marker_pb2.Marker.ADD_MODIFY
    m.parent = ''                    # world-frame pose
    m.layer = 1                      # above default geometry
    m.text = name
    m.pose.position.x = float(x)
    m.pose.position.y = float(y)
    m.pose.position.z = float(z + offset_z)
    m.pose.orientation.w = 1.0
    m.scale.x = m.scale.y = m.scale.z = float(scale)
    m.material.diffuse.r = float(r)
    m.material.diffuse.g = float(g)
    m.material.diffuse.b = float(b)
    m.material.diffuse.a = 1.0
    return m


class DroneLabelsNode(Node):
    def __init__(self):
        super().__init__('drone_labels')

        self.declare_parameter('drone_names', ['drone_0', 'drone_1'])
        self.declare_parameter('targets', [1.0, 5.0, 1.0, -1.0, 5.0, 1.0])
        self.declare_parameter('world_name', WORLD_NAME)
        self.declare_parameter('interval', 0.25)
        self.declare_parameter('label_offset_z', 0.55)
        self.declare_parameter('label_scale', 0.35)

        names = list(self.get_parameter('drone_names')
                     .get_parameter_value().string_array_value)
        self.world_name = (self.get_parameter('world_name')
                           .get_parameter_value().string_value)
        self.interval = (self.get_parameter('interval')
                         .get_parameter_value().double_value)
        self.offset_z = (self.get_parameter('label_offset_z')
                         .get_parameter_value().double_value)
        self.scale = (self.get_parameter('label_scale')
                      .get_parameter_value().double_value)

        self._sources = {name: RealPoseSubscriber(name) for name in names}
        self._names = names
        self._gz = gz.Node()
        self._pub = self._gz.advertise(
            f'/world/{self.world_name}/marker', marker_pb2.Marker)

        self.timer = self.create_timer(self.interval, self._publish)
        self.get_logger().info(
            f'Drone labels active for {len(names)} drones '
            f'(publishing every {self.interval:.2f} s)')

    def _publish(self):
        for i, name in enumerate(self._names):
            pos = self._sources[name].get()
            if pos is None:
                continue
            m = build_label_marker(name, i, pos[0], pos[1], pos[2],
                                   self.offset_z, self.scale)
            self._pub.publish(m)


def main(args=None):
    rclpy.init(args=args)
    node = DroneLabelsNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
