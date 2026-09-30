"""Behaviour test for the net-charge budget in the MPNN redesign stage.

Runs on CPU in about a minute and touches no weights beyond the ProteinMPNN
checkpoints that ship with the package:

    JAX_PLATFORMS=cpu python tests/test_charge_constraint.py

What it checks, on the VHH scaffold this package ships:

1. every sampled design lands exactly on an integer target charge (tolerance 0);
2. every design lands inside the window when a tolerance is given;
3. the target chain never counts toward the budget - only the designed chain does;
4. fixed residues inside the designed chain do count toward it;
5. an unreachable budget is refused with the achievable range, not missed quietly;
6. the cost of conditioning, as the mean per-residue negative log-likelihood the
   model itself reports, is printed next to the unconstrained baseline.

`tests/test_identical_to_upstream.sh` covers the other half of the contract:
with no budget set, the sampler is byte-identical to the base revision.
"""
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bindcraft.model_weights import proteinmpnn_weights
from bindcraft.protein import AMINO_ACIDS, Protein, ResidueFlags
from bindcraft.proteinmpnn import ProteinMPNNSequenceModel

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VHH_SCAFFOLD = os.path.join(PACKAGE_ROOT, 'scaffolds', 'VHH.cif')
TARGET_STRUCTURE = os.path.join(PACKAGE_ROOT, 'settings', 'target', 'structures', 'hPDL1.pdb')
CANDIDATES = 12
SEED = 0

def net_charge(sequence: str) -> int:
    return sum(map(sequence.count, 'KR')) - sum(map(sequence.count, 'DE'))

def read_chain(path: str) -> Protein:
    chains = Protein.from_structure(path)
    return chains[sorted(chains)[0]]

def designed(protein: Protein, designed_residues: slice | None=None) -> Protein:
    flags = protein.flags
    if designed_residues is None:
        return protein.replace(flags=flags | int(ResidueFlags.DESIGN))
    chosen = jnp.zeros(len(flags), dtype=bool).at[designed_residues].set(True)
    return protein.replace(flags=jnp.where(chosen, flags | int(ResidueFlags.DESIGN), flags))

def decode(protein_complex: dict[str, Protein], target_charge=None, tolerance=0.0, temperature=0.1) -> tuple[list[str], list[float]]:
    model = ProteinMPNNSequenceModel(data_dir=proteinmpnn_weights(), temperature=temperature, key=jax.random.PRNGKey(SEED),
                                     variant='neutral', target_charge=target_charge, charge_tolerance=tolerance)
    candidates = model.predict_candidates({'state': protein_complex}, candidate_count=CANDIDATES)
    sequences, scores = [], []
    for candidate in candidates:
        prediction = candidate['state']
        binder = prediction.protein_complex['A']
        sequences.append(''.join(AMINO_ACIDS[index] for index in np.asarray(binder.sequence).argmax(-1)))
        scores.append(float(prediction.metrics['score']))
    return sequences, scores

def report(label: str, sequences: list[str], scores: list[float]) -> list[int]:
    charges = [net_charge(sequence) for sequence in sequences]
    print(f'{label:<34} charge {min(charges):>4} to {max(charges):>4}  mean {np.mean(charges):>6.2f}   NLL {np.mean(scores):.3f}', flush=True)
    return charges

def main() -> int:
    binder = designed(read_chain(VHH_SCAFFOLD))
    alone = {'A': binder}
    with_target = {'A': binder, 'B': read_chain(TARGET_STRUCTURE)}
    failures = []

    baseline, baseline_scores = decode(alone)
    baseline_charges = report('unconstrained', baseline, baseline_scores)
    preferred = int(round(float(np.mean(baseline_charges))))
    for target in (preferred, preferred - 6, preferred + 6):
        sequences, scores = decode(alone, target_charge=target)
        charges = report(f'target {target:+d} exact', sequences, scores)
        if any(charge != target for charge in charges):
            failures.append(f'exact target {target} missed: {sorted(set(charges))}')

    for tolerance in (2, 3):
        target = preferred - 8
        sequences, scores = decode(alone, target_charge=target, tolerance=tolerance)
        charges = report(f'target {target:+d} +-{tolerance}', sequences, scores)
        if any(abs(charge - target) > tolerance for charge in charges):
            failures.append(f'window {target}+-{tolerance} missed: {sorted(set(charges))}')

    target = preferred - 6
    sequences, scores = decode(with_target, target_charge=target)
    charges = report(f'with fixed target chain {target:+d}', sequences, scores)
    if any(charge != target for charge in charges):
        failures.append(f'the designed chain alone should carry the budget, got {sorted(set(charges))}')

    half = len(binder) // 2
    partly_fixed = {'A': designed(read_chain(VHH_SCAFFOLD), slice(0, half))}
    native = ''.join(AMINO_ACIDS[index] for index in np.asarray(partly_fixed['A'].sequence).argmax(-1))
    held_charge = net_charge(native[half:])
    target = preferred - 4
    sequences, scores = decode(partly_fixed, target_charge=target)
    charges = report(f'half fixed (held {held_charge:+d}) {target:+d}', sequences, scores)
    if any(charge != target for charge in charges):
        failures.append(f'fixed residues should count toward the budget, got {sorted(set(charges))}')

    try:
        decode(alone, target_charge=len(binder) + 50)
        failures.append('an unreachable budget was accepted')
    except ValueError as refusal:
        print(f'unreachable budget refused: {refusal}', flush=True)

    for failure in failures:
        print(f'FAIL {failure}', flush=True)
    print('PASS' if not failures else f'{len(failures)} failure(s)', flush=True)
    return 1 if failures else 0

if __name__ == '__main__':
    raise SystemExit(main())
