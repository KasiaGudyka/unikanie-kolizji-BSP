from glob import glob
import os

from setuptools import setup

package_name = 'unikanie_kolizji'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/worlds', glob('worlds/*.world')),
        ('share/' + package_name + '/config', glob('config/*.yaml') + glob('config/*.config')),
        (os.path.join('share', package_name, 'test'), glob('test/*.py')),
    ] + [
        (os.path.join('share', package_name, 'models', d,
                      os.path.relpath(root, os.path.join('models', d))),
         [os.path.join(root, f) for f in files])
        for d in os.listdir('models')
        if os.path.isdir(os.path.join('models', d))
        for root, dirs, files in os.walk(os.path.join('models', d))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Katarzyna Gudyka',
    maintainer_email='katarzyna.gudyka@gmail.com',
    description='Gazebo Harmonic + ROS2 Jazzy drone collision avoidance test package.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'switch_physics_engine=unikanie_kolizji.switch_physics_engine:main',
            'avoidance_mode_select=unikanie_kolizji.avoidance_mode_select:main',
            'world_generator=unikanie_kolizji.world_generator:main',
            'drone_gps=unikanie_kolizji.drone_gps:main',
            'drone_1_gps=unikanie_kolizji.drone_1_gps:main',
            'drone_2_gps=unikanie_kolizji.drone_2_gps:main',
            'drone_collision_avoidance=unikanie_kolizji.drone_collision_avoidance:main',
            'drone_labels=unikanie_kolizji.drone_labels:main',
            'drone_start_gate=unikanie_kolizji.drone_start_gate:main',
            'benchmark_avoidance=unikanie_kolizji.benchmark_avoidance:main',
            'testowy_gui=unikanie_kolizji.sim_gui:main',
        ],
    },
)
