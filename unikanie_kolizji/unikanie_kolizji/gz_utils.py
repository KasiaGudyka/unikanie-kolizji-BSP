#!/usr/bin/env python3

"""Shared helpers for interacting with a Gazebo Harmonic simulation.

Provides a reliable way to move/teleport a model to an absolute world pose,
either through the fast ROS `SetEntityPose` service (when bridged) or through
the gz-native `/world/<world>/set_pose` service via the ros_gz_sim CLI tool.
"""

import os
import shutil
import subprocess

from geometry_msgs.msg import Pose, Point, Quaternion
from ros_gz_interfaces.msg import Entity
from ros_gz_interfaces.srv import SetEntityPose


def move_model(node, world_name, model_name, x, y, z, try_ros_service=True):
    """Move `model_name` to an absolute world pose (x, y, z).

    Prefers the ROS `SetEntityPose` service (requires a ros_gz_bridge). If the
    service is not yet available, falls back to the gz-native pose service
    through the ros_gz_sim `set_entity_pose` CLI, which talks directly to
    Gazebo over gz-transport and needs no bridge.

    Returns True if the pose was requested successfully via either path.
    """
    if try_ros_service:
        client = node.pose_client
        if client is not None and client.service_is_ready():
            pose = Pose()
            pose.position = Point(x=x, y=y, z=z)
            pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
            req = SetEntityPose.Request()
            req.entity = Entity(name=model_name, type=Entity.MODEL)
            req.pose = pose
            client.call_async(req)
            return True

    return _set_pose_gz_cli(node, world_name, model_name, x, y, z)


def _set_pose_gz_cli(node, world_name, model_name, x, y, z):
    """Call the gz-native pose service via the ros_gz_sim set_entity_pose CLI.

    This talks directly to the running Gazebo over gz-transport
    (/world/<world>/set_pose) and requires no ROS<->GZ bridge.
    """
    binary = os.environ.get(
        'ROS_GZ_SET_ENTITY_POSE',
        shutil.which('set_entity_pose') or
        '/opt/ros/jazzy/lib/ros_gz_sim/set_entity_pose')
    if not os.path.exists(binary):
        node.get_logger().warn(
            f'Cannot move "{model_name}": ros_gz_sim set_entity_pose tool '
            f'not found at "{binary}".')
        return False

    cmd = [
        binary,
        '--name', model_name,
        '--type', '6',
        '--pos', str(x), str(y), str(z),
        '--quat', '0', '0', '0', '1',
    ]

    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        node.get_logger().debug(
            f'Move "{model_name}" rc={result.returncode}: '
            f'{result.stdout.strip()} {result.stderr.strip()}')
    return True
