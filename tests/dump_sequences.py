"""Print ProteinMPNN sequences for a fixed scaffold and seed.

Deliberately uses only the API the base revision already had, so the same file
runs in a worktree of any revision and `tests/test_identical_to_upstream.sh` can
diff the output.

    JAX_PLATFORMS=cpu python tests/dump_sequences.py
"""
import os
import sys

import jax
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bindcraft.model_weights import proteinmpnn_weights
from bindcraft.protein import AMINO_ACIDS, Protein, ResidueFlags
from bindcraft.proteinmpnn import ProteinMPNNSequenceModel

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VHH_SCAFFOLD = os.path.join(PACKAGE_ROOT, 'scaffolds', 'VHH.cif')
CANDIDATES = 12
SEED = 0

def main() -> int:
    chains = Protein.from_structure(VHH_SCAFFOLD)
    binder = chains[sorted(chains)[0]]
    protein_complex = {'A': binder.replace(flags=binder.flags | int(ResidueFlags.DESIGN))}
    model = ProteinMPNNSequenceModel(data_dir=proteinmpnn_weights(), temperature=0.1, key=jax.random.PRNGKey(SEED), variant='neutral')
    for candidate in model.predict_candidates({'state': protein_complex}, candidate_count=CANDIDATES):
        prediction = candidate['state']
        sequence = ''.join(AMINO_ACIDS[index] for index in np.asarray(prediction.protein_complex['A'].sequence).argmax(-1))
        print(f"{sequence}\t{float(prediction.metrics['score']):.6f}\t{float(prediction.metrics['seqid']):.6f}")
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
