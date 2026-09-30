"""Net-charge constrained decoding for the ProteinMPNN sampler.

The controller is a port of the one in the LigandMPNN fork at
https://github.com/profdocpizza/LigandMPNN (branch ``feature/constrained-decoding``,
``constraints.py`` / ``docs/CONSTRAINED_DECODING.md``, MIT licensed) to the JAX
sampler shipped with BindCraft2. Two stages act on the logits of every designed
step, after the amino-acid bias and before the Gumbel draw:

* **lookahead reweighting** multiplies the model's own distribution by the
  probability that the steps *after* this one can still supply whatever the
  target demands, under a normal approximation to that remaining sum whose
  per-position moments are read off the model's marginals as decoding proceeds.
  This conditions the endpoint and nothing else, which is what a charge budget
  asks for; an earlier controller in the reference implementation instead pinned
  the expected contribution of every step and cost roughly ten times as much
  likelihood for the same endpoint.
* **hard reachability pruning** masks any amino acid whose selection would put
  the target out of reach of the undecoded steps. This, not the reweighting, is
  what makes the budget a guarantee rather than a bias. The mask is dropped for
  a step where it would leave nothing to sample.

Definitions, following the reference implementation so the number means the same
thing in both: Asp and Glu count -1, Lys and Arg count +1, **histidine counts 0**
(its charge at the reported pH is partial and context dependent) and free termini
are not counted. That is the integer ``Binder_Net_Charge`` of ``rank.py``
(``K+R minus D+E``), not the pH-dependent ``Binder_Net_Charge`` filter metric,
which is a continuous Henderson-Hasselbalch reading of the same sequence.

Scope is every resolved residue of a chain that holds at least one designed
position, so a fixed residue inside the binder still counts toward its charge and
the target chains - which hold no designed positions - never do. A tied group
contributes **once**, because the group is one physical residue decoded in
several states or copies.

Reachability is computed as an interval, which is exact for a charge in
``{-1, 0, +1}`` as long as every designed step keeps one uncharged amino acid in
its alphabet. A bias that omits all fifteen uncharged amino acids would leave a
lattice the interval does not describe; ``proteinmpnn.py`` refuses that case
rather than silently missing the target.

The sampler is untouched when no target charge is set: the schedule is built only
when one is, and the branch is resolved at trace time.
"""
import jax
import jax.numpy as jnp

MPNN_ALPHABET = 'ACDEFGHIKLMNPQRSTVWYX'
FORMAL_CHARGE = {'D': -1.0, 'E': -1.0, 'K': 1.0, 'R': 1.0}
CHARGE_PER_AMINO_ACID = tuple(FORMAL_CHARGE.get(amino_acid, 0.0) for amino_acid in MPNN_ALPHABET)
UNREACHABLE_LOGIT = -1000000000.0
LOOKAHEAD_FLOOR = -50.0
VARIANCE_FLOOR = 1e-08
DEFAULT_CHARGE_LAMBDA_MAX = 8.0

def charge_values() -> jnp.ndarray:
    return jnp.asarray(CHARGE_PER_AMINO_ACID, dtype=jnp.float32)

def designed_chain_scope(chain_indices: jnp.ndarray, resolved: jnp.ndarray, designed: jnp.ndarray) -> jnp.ndarray:
    """Resolved residues of every chain that holds at least one designed position."""
    chain_membership = jax.nn.one_hot(chain_indices, chain_indices.shape[0])
    chain_designs = (chain_membership * designed[:, None]).max(0)
    return (chain_membership * chain_designs[None, :]).sum(-1) * resolved

def _totals_after_each_step(step_values: jnp.ndarray) -> jnp.ndarray:
    """``[k]`` = the sum of ``step_values`` over every step later than ``k``."""
    reversed_totals = jnp.cumsum(step_values[::-1])[::-1]
    return jnp.concatenate([reversed_totals, jnp.zeros((1,), step_values.dtype)])[1:]

def charge_schedule(decoding_order: jnp.ndarray, scope: jnp.ndarray, designed: jnp.ndarray, native_charge: jnp.ndarray, allowed_amino_acids: jnp.ndarray, values: jnp.ndarray) -> dict[str, jnp.ndarray]:
    """Per-decoding-step arrays the controller reads, in decoding order.

    ``decoding_order`` is ``[K]`` or ``[K, G]`` for tied groups; ``scope``,
    ``designed`` and ``native_charge`` are ``[L]``; ``allowed_amino_acids`` is
    ``[L, 21]``, 1 where the bias has not omitted that amino acid.
    """
    positions = decoding_order if decoding_order.ndim == 2 else decoding_order[:, None]
    in_scope = scope[positions]
    representative = jnp.take_along_axis(positions, jnp.argmax(in_scope, axis=-1)[:, None], axis=-1)[:, 0]
    multiplicity = in_scope.max(-1)  #a tied group is one physical residue, so it counts once
    designed_step = (designed[positions] * in_scope).max(-1)
    group_allowed = jnp.where(in_scope[..., None] > 0, allowed_amino_acids[positions], jnp.ones_like(allowed_amino_acids[positions]))
    step_allowed = group_allowed.min(1)  #the intersection over the tied positions
    native_step = native_charge[representative] * multiplicity
    fixed_step = jnp.where(designed_step > 0, 0.0, native_step)
    lowest_allowed = jnp.min(jnp.where(step_allowed[:, :20] > 0, values[None, :20], jnp.inf), axis=-1)
    highest_allowed = jnp.max(jnp.where(step_allowed[:, :20] > 0, values[None, :20], -jnp.inf), axis=-1)
    reachable_low = jnp.where(designed_step > 0, multiplicity * lowest_allowed, fixed_step)
    reachable_high = jnp.where(designed_step > 0, multiplicity * highest_allowed, fixed_step)
    designed_count = jnp.where(designed_step > 0, multiplicity, 0.0)
    return {'multiplicity': multiplicity, 'designed': designed_step, 'allowed': step_allowed,
            'remaining_low': _totals_after_each_step(reachable_low), 'remaining_high': _totals_after_each_step(reachable_high),
            'remaining_fixed': _totals_after_each_step(fixed_step), 'remaining_designed': _totals_after_each_step(designed_count),
            'remaining_designed_squared': _totals_after_each_step(designed_count ** 2)}

def charge_state() -> dict[str, jnp.ndarray]:
    return {'charge': jnp.zeros(()), 'marginal_mean': jnp.zeros(()), 'marginal_variance': jnp.zeros(()), 'observations': jnp.zeros(())}

def charge_logit_delta(state: dict, step: dict, sampling_logits: jnp.ndarray, constraint: dict) -> tuple[jnp.ndarray, dict]:
    """This step's additive logit term, with the updated moment estimates.

    ``sampling_logits`` is ``[1, 21]`` after the bias and the temperature
    division, so the lookahead term is a log-probability correction added at the
    scale it is sampled on.
    """
    values, target = constraint['values'], constraint['target']
    tolerance, lambda_max = constraint['tolerance'], constraint['lambda_max']
    active = step['designed'] * (step['multiplicity'] > 0)
    marginal = jax.nn.softmax(sampling_logits[0, :20])
    step_mean = (marginal * values[:20]).sum()
    step_variance = jnp.clip((marginal * values[:20] ** 2).sum() - step_mean ** 2, 0.0, None)
    observations = state['observations'] + active
    marginal_mean = state['marginal_mean'] + active * step_mean
    marginal_variance = state['marginal_variance'] + active * step_variance
    observed = jnp.maximum(observations, 1.0)
    mean_per_position = marginal_mean / observed
    variance_per_position = jnp.clip(marginal_variance / observed, VARIANCE_FLOOR, None)
    deficit = target - state['charge'] - step['remaining_fixed']
    requirement = deficit - step['multiplicity'] * values[:20]
    expected_remaining = step['remaining_designed'] * mean_per_position
    spread = jnp.sqrt(jnp.clip(step['remaining_designed_squared'] * variance_per_position, VARIANCE_FLOOR, None))
    #a deadband: no penalty while the remainder is expected to cover the requirement to within the tolerance
    distance = jnp.clip(jnp.abs(requirement - expected_remaining) - tolerance, 0.0, None)
    log_reachable = jnp.clip(-0.5 * (distance / spread) ** 2, LOOKAHEAD_FLOOR, None)
    log_reachable = log_reachable - log_reachable.max()
    #the last designed step has no remainder to look ahead to, and the mask alone decides it
    log_reachable = jnp.where(step['remaining_designed'] > 0.5, log_reachable, 0.0)
    steer = jnp.clip(log_reachable, -lambda_max, 0.0)
    contribution = step['multiplicity'] * values
    lowest_end = state['charge'] + step['remaining_low'] + contribution
    highest_end = state['charge'] + step['remaining_high'] + contribution
    reachable = (lowest_end <= target + tolerance) & (highest_end >= target - tolerance)
    reachable = reachable & (step['allowed'] > 0) & (jnp.arange(values.shape[0]) < 20)
    #pruning that would leave nothing to sample is dropped rather than emitting a degenerate distribution
    mask = jnp.where(reachable.any(), jnp.where(reachable, 0.0, UNREACHABLE_LOGIT), jnp.zeros_like(values))
    delta = jnp.zeros_like(values).at[:20].set(steer) + mask
    return (active * delta[None, :], {'charge': state['charge'], 'marginal_mean': marginal_mean, 'marginal_variance': marginal_variance, 'observations': observations})

def charge_commit(state: dict, step: dict, sampled_amino_acid: jnp.ndarray, values: jnp.ndarray) -> dict:
    """Record the residue the step settled on; a fixed step arrives here already forced to its native one."""
    return {**state, 'charge': state['charge'] + step['multiplicity'] * (sampled_amino_acid * values).sum()}

def scope_charge_bounds(scope, designed, native_charge, allowed_amino_acids, values) -> tuple[float, float, int]:
    """The charges this scope can reach and how many designed positions it holds, over concrete arrays."""
    import numpy as np
    scope, designed = np.asarray(scope, dtype=np.float32), np.asarray(designed, dtype=np.float32)
    native_charge, allowed_amino_acids, values = np.asarray(native_charge), np.asarray(allowed_amino_acids), np.asarray(values)
    fixed_total = float((native_charge * scope * (1.0 - designed)).sum())
    designed_positions = (scope * designed) > 0
    if not designed_positions.any():
        return (fixed_total, fixed_total, 0)
    allowed_values = np.where(allowed_amino_acids[designed_positions][:, :20] > 0, values[None, :20], np.nan)
    return (fixed_total + float(np.nanmin(allowed_values, axis=-1).sum()), fixed_total + float(np.nanmax(allowed_values, axis=-1).sum()), int(designed_positions.sum()))

def uncharged_amino_acid_missing(allowed_amino_acids, values, positions) -> bool:
    """True when a designed position's alphabet holds no uncharged amino acid, which is what the interval relaxation needs."""
    import numpy as np
    allowed_amino_acids, values = np.asarray(allowed_amino_acids), np.asarray(values)
    considered = allowed_amino_acids[np.asarray(positions) > 0]
    if considered.size == 0:
        return False
    return not bool(((considered[:, :20] > 0) & (np.abs(values[None, :20]) < 1e-12)).any(-1).all())
