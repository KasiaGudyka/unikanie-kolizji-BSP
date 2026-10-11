#!/usr/bin/env python3

import argparse


AVAILABLE_MODES = ['method_1', 'method_2', 'method_3',
                   'method_4', 'method_5', 'method_6']


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Select the collision avoidance method for the drone test scenario.')
    parser.add_argument(
        '--mode',
        choices=AVAILABLE_MODES,
        default='method_2',
        help='Collision avoidance method. method_2 (3D-SWAP roundabout) is the '
             'recommended default for crossing paths; method_1 (GPS/no-fly-zone '
             'repulsion) is also implemented; method_3 is the DPSO trajectory '
             'planner; method_4 convex hull; method_5 Gaussian mixture; '
             'method_6 fuses GMM perception with STGO swarm-consensus '
             '(Peng et al., IROS 2022).')
    args = parser.parse_args()

    print(f'Collision avoidance mode selected: {args.mode}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
