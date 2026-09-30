"""What the net-charge budget costs in model likelihood, measured on three globular backbones.

Two questions, one run:

1. **What does the ask cost?** Decode designs under a budget and compare their mean per-residue
   negative log-likelihood with unconstrained decoding on the same backbone.
2. **How much of that is the controller?** Rejection sampling realises p(sequence | charge = c)
   exactly: draw a large unconstrained pool and keep the designs that already have charge c. Those
   come from the true conditional, so their mean NLL is the honest reference, and anything the
   constrained sampler spends above it is the controller's own overhead rather than the price of
   asking. This is the yardstick used by the reference implementation in profdocpizza/LigandMPNN
   (benchmarks/bench_rejection.py).

Targets are set relative to each backbone's own unconstrained mean charge, because an absolute
charge is not comparable across sizes: a 10 e shift is a large ask on a 54-mer and a small one on
a 214-mer.

    JAX_PLATFORMS=cpu python benchmarks/charge_cost.py --pool 128 --n-designs 16   # quick
    python benchmarks/charge_cost.py                                               # the full run

Writes benchmarks/results/charge_cost_designs.csv (every design) and
benchmarks/results/charge_cost_summary.csv (per backbone and target). Structures are fetched from
RCSB into benchmarks/structures/ on demand and are not kept in the repository.
"""
import argparse
import csv
import os
import statistics
import sys
import urllib.request

import jax
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bindcraft.model_weights import proteinmpnn_weights
from bindcraft.protein import AMINO_ACIDS, Protein, ResidueFlags
from bindcraft.proteinmpnn import ProteinMPNNSequenceModel

HERE = os.path.dirname(os.path.abspath(__file__))
STRUCTURES = os.path.join(HERE, 'structures')
RESULTS = os.path.join(HERE, 'results')
RCSB = 'https://files.rcsb.org/download/{entry}.pdb'
#one small, one medium, one large single-domain globular protein
PANEL = (('1ENH', 'A', 'engrailed homeodomain'), ('1BRS', 'A', 'barnase'), ('1AKE', 'A', 'adenylate kinase'))
TEMPERATURE = 0.1
SEED = 90210
CHUNK = 32

def fetch(entry: str) -> str:
    os.makedirs(STRUCTURES, exist_ok=True)
    path = os.path.join(STRUCTURES, f'{entry}.pdb')
    if not os.path.exists(path):
        urllib.request.urlretrieve(RCSB.format(entry=entry), path)
    return path

def net_charge(sequence: str) -> int:
    return sum(map(sequence.count, 'KR')) - sum(map(sequence.count, 'DE'))

def designed_chain(entry: str, chain: str) -> Protein:
    protein = Protein.from_structure(fetch(entry), chains=chain)[chain]
    return protein.replace(flags=protein.flags | int(ResidueFlags.DESIGN))

def decode(protein: Protein, count: int, target_charge=None, seed: int=SEED) -> list[dict]:
    model = ProteinMPNNSequenceModel(data_dir=proteinmpnn_weights(), temperature=TEMPERATURE,
                                     key=jax.random.PRNGKey(seed), variant='neutral', target_charge=target_charge)
    designs, drawn = [], 0
    while drawn < count:
        batch = min(CHUNK, count - drawn)
        for candidate in model.predict_candidates({'state': {'A': protein}}, candidate_count=batch):
            prediction = candidate['state']
            sequence = ''.join(AMINO_ACIDS[index] for index in np.asarray(prediction.protein_complex['A'].sequence).argmax(-1))
            designs.append({'sequence': sequence, 'charge': net_charge(sequence), 'nll': float(prediction.metrics['score'])})
        drawn += batch
    return designs

def mean(values) -> float:
    values = list(values)
    return float(statistics.fmean(values)) if values else float('nan')

def spread(values) -> float:
    values = list(values)
    return float(statistics.stdev(values)) if len(values) > 1 else float('nan')

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--pool', type=int, default=512, help='unconstrained designs per backbone, for the rejection reference')
    parser.add_argument('--n-designs', type=int, default=32, help='constrained designs per target')
    parser.add_argument('--min-bin', type=int, default=10, help='pool designs needed at a charge before it is used as a reference')
    parser.add_argument('--offsets', type=int, nargs='*', default=[-12, -8, -6, -4, -2, 0, 2, 4, 6, 8, 12],
                        help="targets, as offsets from the backbone's own unconstrained mean charge")
    arguments = parser.parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    design_rows, summary_rows = [], []

    for entry, chain, description in PANEL:
        protein = designed_chain(entry, chain)
        length = len(protein)
        pool = decode(protein, arguments.pool)
        pool_charges = [design['charge'] for design in pool]
        pool_nll = mean(d['nll'] for d in pool)
        preferred = int(round(mean(pool_charges)))
        print(f'\n=== {entry} {description}  L={length}  pool {len(pool)}  '
              f'charge {min(pool_charges):+d} to {max(pool_charges):+d} (mean {mean(pool_charges):+.2f})  '
              f'NLL {pool_nll:.4f}', flush=True)
        for design in pool:
            design_rows.append({'backbone': entry, 'length': length, 'condition': 'unconstrained',
                                'target': '', 'charge': design['charge'], 'nll': design['nll']})
        pool_at = {}
        for design in pool:
            pool_at.setdefault(design['charge'], []).append(design['nll'])

        for offset in arguments.offsets:
            target = preferred + offset
            designs = decode(protein, arguments.n_designs, target_charge=target)
            charges = [design['charge'] for design in designs]
            constrained_nll = mean(d['nll'] for d in designs)
            for design in designs:
                design_rows.append({'backbone': entry, 'length': length, 'condition': 'constrained',
                                    'target': target, 'charge': design['charge'], 'nll': design['nll']})
            reference = pool_at.get(target, [])
            enough = len(reference) >= arguments.min_bin
            row = {'backbone': entry, 'description': description, 'length': length, 'preferred_charge': preferred,
                   'offset': offset, 'target': target, 'demand_per_residue': abs(offset) / length,
                   'n_constrained': len(designs), 'on_target': sum(charge == target for charge in charges),
                   'constrained_nll': constrained_nll, 'constrained_sd': spread(d['nll'] for d in designs),
                   'pool_nll': pool_nll, 'pool_sd': spread(d['nll'] for d in pool),
                   'n_rejection': len(reference) if enough else 0,
                   'rejection_nll': mean(reference) if enough else '',
                   'rejection_sd': spread(reference) if enough else '',
                   'excess_nll': (constrained_nll - mean(reference)) if enough else '',
                   'cost_vs_pool': constrained_nll - pool_nll}
            summary_rows.append(row)
            note = (f'   rejection {row["rejection_nll"]:.4f} (n={row["n_rejection"]})   excess {row["excess_nll"]:+.4f}'
                    if enough else '   rejection n/a')
            print(f'  target {target:+4d} (offset {offset:+3d})  on target {row["on_target"]:2d}/{len(designs)}'
                  f'   NLL {constrained_nll:.4f}   cost {row["cost_vs_pool"]:+.4f}{note}', flush=True)

    for name, rows in (('charge_cost_designs.csv', design_rows), ('charge_cost_summary.csv', summary_rows)):
        with open(os.path.join(RESULTS, name), 'w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f'wrote {os.path.join(RESULTS, name)} ({len(rows)} rows)', flush=True)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
