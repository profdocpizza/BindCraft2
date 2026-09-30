# A net-charge budget for the redesign stage

This is [profdocpizza/BindCraft2](https://github.com/profdocpizza/BindCraft2), a fork of
[PacesaLab/BindCraft2](https://github.com/PacesaLab/BindCraft2) that tracks upstream and adds one
feature: the ProteinMPNN redesign stage can be told what net charge its sequences must carry.

Upstream has `mpnn_variant` (`neutral` / `negative` / `positive`) and `aa_bias`, which lean on the
composition but bound nothing, so the only way to get a charge was to generate designs and discard
the ones that missed - after paying to fold and filter them. This makes the charge a property of
every sequence the stage emits, decided while it decodes.

Base: upstream `a8d0f20` (2026-09-29). Measurements below were taken on one RTX 5090 (32 GB,
sm_120), Python 3.12, jax 0.11.2.

## Settings

| Setting | Default | Meaning |
|---|---|---|
| `mpnn_target_charge` | `null` | Net charge every redesigned sequence must carry. `null` leaves the sampler exactly as upstream. |
| `mpnn_charge_tolerance` | `0` | Half-width of the accepted window. `0` asks for the exact integer. |
| `mpnn_charge_lambda_max` | `8.0` | Cap in logit units on the soft steering that precedes the hard mask. Raise it only to diagnose. |

```json
{
  "target": "hPDL1",
  "modality": "binder",
  "mpnn_target_charge": -8,
  "mpnn_charge_tolerance": 3
}
```

That asks for a net charge in `[-11, -5]` on every candidate the redesign stage emits. All three
settings take part in the design-identity hash, and all three are campaign-wide: a parameter sweep
cannot vary them per arm, because the MPNN model object is built once for the campaign.

## What it guarantees, and over what

**Net charge** counts Asp and Glu as −1, Lys and Arg as +1. **Histidine counts 0** — its charge at
the reported pH is partial and context dependent — and **free termini are not counted**. Both
choices keep the target an integer, which is what makes exactness meaningful. This is the same
quantity `rank.py` reports as `Binder_Net_Charge` (`K+R minus D+E`); it is *not* the pH-dependent
`Binder_Net_Charge` filter metric, which is a continuous Henderson-Hasselbalch reading of the same
sequence and will differ by a fraction of a charge unit.

**Scope** is every resolved residue of a chain that holds at least one designed position. So:

* the target chains never count — they hold no designed positions;
* a **fixed** residue inside the binder does count, which is what "this binder should have net
  charge −8" normally means, and matters when `redesign_interface` is false and the interface is
  held;
* a **tied** group contributes **once**, not once per state or copy, because the group is one
  physical residue decoded in several places. Multistate campaigns (one binder against human and
  mouse targets, say) therefore get the binder's charge, not twice it.

## How it works

No weights are retrained and no checkpoint is touched. At each autoregressive step the sampler
emits 21 amino-acid logits; two deterministic stages act on them, after the amino-acid bias and
before the Gumbel draw:

* **Lookahead reweighting.** The exact conditional is
  `p(s_t | s_<t, target) = p(s_t | s_<t) · P(target reachable | s_<=t) / Z`, so the logits are
  multiplied by the probability that the positions after this one can still supply whatever the
  target demands, estimated by a normal approximation to that remaining sum whose per-position mean
  and variance are read off the model's own marginals as decoding proceeds. This conditions the
  **endpoint** and nothing else.
* **Hard reachability pruning.** Any amino acid whose selection would put the target out of reach
  of the undecoded steps is masked. This, not the reweighting, is what makes the budget a guarantee.
  Pruning that would leave nothing to sample is dropped rather than emitting a degenerate
  distribution.

`log_probs`, and therefore the `score` each candidate reports, are computed from the **raw** logits,
so the number stays comparable between constrained and unconstrained runs.

An unreachable budget is refused before decoding, naming the achievable range:

```
ValueError: mpnn_target_charge 177 +-0 is unreachable: 127 designed position(s) over the
designed chains reach a net charge of -127 to 127
```

## What it costs

`tests/test_charge_constraint.py`, on the VHH scaffold this package ships (127 designed positions,
12 designs per row, temperature 0.1, `neutral` weights, CPU):

| condition | achieved charge | mean per-residue NLL |
|---|---|---|
| unconstrained | −5 to +1 (mean −1.8) | 0.726 |
| `mpnn_target_charge: -2` (its own preference) | −2 to −2 | 0.762 |
| `mpnn_target_charge: -8` | −8 to −8 | 0.798 |
| `mpnn_target_charge: 4` | +4 to +4 | 0.827 |
| `-10`, tolerance 2 | −10 to −8 | 0.795 |
| `-10`, tolerance 3 | −7 to −7 | 0.787 |
| `-8` with a fixed target chain in the complex | −8 to −8 | 1.213 |
| `-6` with half the binder held fixed | −6 to −6 | 0.976 |

Conditioning at the charge the model already prefers costs about +0.04 nats per residue, and a 6 e
shift about +0.07 to +0.10. The reference implementation's benchmark over 17 backbones reports the
same magnitude (+0.06 nats at the preferred charge, rising with the size of the shift).

Two things the table shows that are worth reading twice. The row with a fixed target chain has a
much higher NLL (1.213) because the score averages over every resolved residue including the
target's, not because the budget is expensive. And the tolerance rows land on the window edge
nearest the model's own preference rather than spreading across the window — the tolerance is a
deadband, so inside it the model is left alone and only the edge is enforced. Use the tolerance to
buy likelihood back, not to get a spread of charges.

## What is not claimed

* **No folding, expression or synthesis validation.** The cost above is the model's own likelihood,
  a cheap proxy for design quality rather than evidence of it. Whether a charge-shifted design still
  refolds to its backbone is what the campaign's own refold and filter stages are for.
* **Charge is not pI.** At a fixed net charge the isoelectric point still varies by more than a pH
  unit, because pI depends on the counts of ionisable residues and not only on their difference.
* **Reachability is an interval.** That is exact for charges in `{-1, 0, +1}` as long as every
  designed position keeps one uncharged amino acid in its alphabet. A bias or `omit` list that
  removes all fifteen uncharged amino acids somewhere would leave a lattice the interval does not
  describe; that case is refused at setup rather than missed silently.
* **Diversity under the budget is not characterised** beyond the composition shift implied above.

## Provenance

The controller is a port of the one in
[profdocpizza/LigandMPNN](https://github.com/profdocpizza/LigandMPNN), branch
`feature/constrained-decoding` (`constraints.py`, `docs/CONSTRAINED_DECODING.md`; LigandMPNN is MIT
licensed). That implementation is PyTorch and sits in the LigandMPNN codebase; **BindCraft2 does not
use LigandMPNN.** It ships its own JAX/Haiku ProteinMPNN in `bindcraft/mpnn/` with its own
checkpoints in `bindcraft/weights/`, so the two-stage controller was re-implemented against
`bindcraft/mpnn/sample.py` rather than vendored. Per-step schedules are built in
`bindcraft/proteinmpnn.py` and carried through `hk.scan` as scan inputs; the controller itself is
`bindcraft/mpnn/constraints.py`.

The reference implementation's other two constraints — an ε₂₈₀ floor and SPPS-motif steering — were
not ported. Both would fit the same machinery (ε₂₈₀ is another linear functional; SPPS risk is not
one and carries no guarantee), and neither is needed by the campaigns this fork runs.

## Tests

```bash
JAX_PLATFORMS=cpu python tests/test_charge_constraint.py        # the table above, ~1 min, CPU
bash tests/test_identical_to_upstream.sh upstream/main          # no budget set => byte-identical
```

The second test decodes a fixed scaffold at a fixed seed in a worktree of the base revision and in
the working tree, and diffs the sequences and scores byte for byte. It passes: with
`mpnn_target_charge` unset, the schedule is never built and the branch is resolved at trace time, so
the sampler is the upstream sampler.

---

## Licensing

BindCraft2's source-available licence permits modification and derivative works, with prominent
notice of modifications; this page is that notice. The controller is ported from
profdocpizza/LigandMPNN (MIT), attributed here and in `bindcraft/mpnn/constraints.py`.
