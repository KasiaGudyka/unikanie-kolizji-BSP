#!/usr/bin/env python3

"""Method 6: GMM perception + STGO swarm-consensus trajectory planner.

Pure implementation (no ROS / Gazebo dependencies).  Method 6 builds on the
DPSO trajectory planner (method 3) and fuses one perceptual cost term from
method 5:

  * a Gaussian-mixture density potential  p(x) = sum exp(-||x-mu_k||^2/2 sigma^2)
    over the neighbours, penalising planned positions inside the densest part
    of the surrounding swarm;

and one consensus mechanism from the paper (Peng et al., IROS 2022):

  * STGO (Swarm Trajectory Global Optimal): after every drone has run its
    local DPSO independently, the globally cheapest gbest plan (lowest cost
    across the swarm) is selected and stored on every drone as the consensus
    reference.  On the next replan that adopted plan anchors the whole swarm:
    each drone treats it as a fully committed dynamic obstacle (keep-out
    weight 1.0), while still driving toward its own goal.  This breaks the
    symmetric local minima that plague per-drone optimisation and produces
    strict consensus on future occupancy without forcing identical velocities
    on drones with different goals.

The convex-hull membership penalty (old method 6) is removed — it does not
appear in the paper and is available standalone as method 4.
"""

import dataclasses
from typing import Dict, List, Optional, Sequence, Tuple

from . import dpso_swarm
from .gmm_avoid import gmm_density_gradient

Vec3 = Tuple[float, float, float]


@dataclasses.dataclass
class CombinedParams:
    """Tunables of the GMM+STGO method (features DPSOParams + extras)."""

    dpso: dpso_swarm.DPSOParams = dataclasses.field(
        default_factory=dpso_swarm.DPSOParams)
    gmm_sigma: float = 0.8    # kernel width for the GMM density potential
    comb_w_gmm: float = 1.4   # J weight of the GMM density potential
    stgo_adoption: float = 1.0  # keep-out weight of the adopted plan (0..1)


def _extra_cost(others, params: CombinedParams):
    """Return ``extra_cost(pos, t)`` for DPSOParams given the neighbours."""
    def cost(pos: Tuple[float, float, float], t: int):
        density, _, _ = gmm_density_gradient(pos, others, params.gmm_sigma)
        return params.comb_w_gmm * density

    return cost


# --- public entry points -----------------------------------------------------

def method_6_combined(own, others, own_goal, params=None, state=None,
                      others_vel=None, external_gbests=None,
                      external_weights=None, self_id=None):
    """GMM+DPSO local plan for the ego drone (pass 1 of STGO).

    ``own`` / ``others`` / ``own_goal`` are 3-tuples in world metres.
    ``self_id`` is the drone's own id; it seeds the swarm with the adopted
    consensus plan when that plan was produced by another drone, so the
    locally cheapest manoeuvre is always present in the population.
    Returns ``(vx, vy, vz, new_state)`` like :func:`dpso_swarm.method_3_dpso`.
    """
    params = params or CombinedParams()
    dp = params.dpso
    dp = dataclasses.replace(dp, extra_cost=_extra_cost(others, params))
    seed = None
    if state is not None and getattr(state, 'consensus_gbest', None) \
            is not None:
        owner = getattr(state, 'consensus_owner', None)
        if owner is not None and owner != self_id:
            seed = state.consensus_gbest
    return dpso_swarm.method_3_dpso(own, others, own_goal, dp, state,
                                    others_vel=others_vel,
                                    external_gbests=external_gbests,
                                    external_weights=external_weights,
                                    extra_seed=seed)


def stgo_consensus(local_plans: Dict[str, Tuple[List[float], float]]
                   ) -> Optional[str]:
    """Return the drone_id whose gbest has the lowest cost, or ``None``.

    ``local_plans`` maps drone_id to ``(gbest, cost)`` from pass 1.
    This is the single centralised ``min()`` call — equivalent to the paper's
    "broadcast P_{i,gbest} ... the result with the lowest cost will be chosen
    as the final trajectories for all UAVs."
    """
    if not local_plans:
        return None
    return min(local_plans, key=lambda k: local_plans[k][1])


def method_6_prepare(state, adopted_gbest, params=None, adoption_weight=1.0,
                     owner=None):
    """Store the STGO-consensus gbest on the drone's state (pass 2).

    The drone keeps executing its own local gbest this tick (different
    drones have different goals, so identical velocities would be wrong).
    The adopted plan is remembered as a committed reference owned by the
    winner drone; on the next replan non-owner drones avoid it with a full
    keep-out weight while the owner follows its own plan (no self-model).

    Returns the (possibly new) state.
    """
    params = params or CombinedParams()
    weight = adoption_weight if adoption_weight is not None \
        else params.stgo_adoption
    if state is None:
        filled = list(adopted_gbest) if adopted_gbest is not None else None
        state = dpso_swarm.DPSOState(
            particles=[], gbest=filled or [0.0] * (3 * params.dpso.horizon),
            gbest_cost=1e18)
    if adopted_gbest is not None and weight >= 1.0:
        state.consensus_gbest = list(adopted_gbest)
    elif adopted_gbest is not None and weight > 0.0:
        prev = state.consensus_gbest or adopted_gbest
        state.consensus_gbest = [
            weight * adopted_gbest[i] + (1.0 - weight) * prev[i]
            for i in range(min(len(prev), len(adopted_gbest)))]
    else:
        state.consensus_gbest = None
    state.consensus_owner = owner
    return state