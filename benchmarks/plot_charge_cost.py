"""Figure: what the net-charge budget costs in model likelihood.

    python benchmarks/plot_charge_cost.py     # reads results/charge_cost_summary.csv

Panel a is the cost of the ask - mean per-residue NLL above unconstrained decoding on the same
backbone - against the size of the shift demanded, with the true conditional (rejection sampling
from the unconstrained pool) as open markers wherever the pool supplied enough designs at that
exact charge. Panel b is the vertical gap between the two: the controller's own overhead.
"""
import argparse
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
#Okabe-Ito: distinguishable under deuteranopia, no red/green opposition
COLOURS = {'1ENH': '#0072B2', '1BRS': '#E69F00', '1AKE': '#009E73'}
SIZES = (8, 7, 6)

def style() -> None:
    base, small, tiny = SIZES
    mpl.rcParams.update({'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
                         'font.size': base, 'axes.titlesize': base, 'axes.labelsize': base,
                         'legend.fontsize': small, 'xtick.labelsize': tiny, 'ytick.labelsize': tiny,
                         'axes.titlelocation': 'left', 'axes.titlepad': 6, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6,
                         'ytick.major.width': 0.6, 'lines.linewidth': 1.2, 'legend.frameon': False})

def confidence_half_width(spread, count):
    return stats.t.ppf(0.975, count - 1) * spread / count ** 0.5

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary', default=os.path.join(HERE, 'results', 'charge_cost_summary.csv'))
    parser.add_argument('--out', default=os.path.normpath(os.path.join(HERE, '..', 'docs', 'source', '_static', 'charge_cost.png')))
    arguments = parser.parse_args()
    style()
    summary = pd.read_csv(arguments.summary)
    order = sorted(summary.backbone.unique(), key=lambda name: summary.loc[summary.backbone == name, 'length'].iloc[0])

    figure, (cost_axes, overhead_axes) = plt.subplots(1, 2, figsize=(7.0, 2.9), gridspec_kw={'width_ratios': [1.55, 1.0]})

    for backbone in order:
        rows = summary[summary.backbone == backbone].sort_values('offset')
        colour = COLOURS[backbone]
        error = [confidence_half_width(sd, n) for sd, n in zip(rows.constrained_sd, rows.n_constrained)]
        cost_axes.errorbar(rows.offset, rows.cost_vs_pool, yerr=error, color=colour, marker='o',
                           markersize=3.2, elinewidth=0.6, capsize=1.6, zorder=3)
        reference = rows.dropna(subset=['excess_nll'])
        reference_cost = reference.rejection_nll - reference.pool_nll
        reference_error = [confidence_half_width(sd, n) for sd, n in zip(reference.rejection_sd, reference.n_rejection)]
        cost_axes.errorbar(reference.offset, reference_cost, yerr=reference_error, color=colour, marker='o',
                           markersize=4.2, markerfacecolor='white', markeredgewidth=0.9, linestyle='none',
                           elinewidth=0.6, capsize=1.6, zorder=4)
        last = rows.iloc[-1]
        cost_axes.annotate(f'{backbone} ({last.length} aa)', xy=(last.offset, last.cost_vs_pool),
                           xytext=(3, 0), textcoords='offset points', color=colour, va='center', fontsize=SIZES[1])

    cost_axes.axhline(0.0, color='#999999', linewidth=0.6, zorder=1)
    cost_axes.set_xlabel('net charge asked for, relative to what the model produces unprompted (e)')
    cost_axes.set_ylabel('cost (nats per residue)')
    cost_axes.set_title('a  The ask: likelihood given up against unconstrained decoding', loc='left', fontweight='normal')
    cost_axes.set_xticks(sorted(summary.offset.unique()))
    cost_axes.margins(x=0.13, y=0.08)
    cost_axes.plot([], [], color='#444444', marker='o', markersize=3.2, label='decoded under the budget')
    cost_axes.plot([], [], color='#444444', marker='o', markersize=4.2, markerfacecolor='white',
                   markeredgewidth=0.9, linestyle='none', label='exact conditioning (rejection sampling)')
    cost_axes.legend(loc='upper center', fontsize=SIZES[1], handlelength=1.6, borderaxespad=0.2)

    for position, backbone in enumerate(order):
        rows = summary[summary.backbone == backbone].dropna(subset=['excess_nll'])
        colour = COLOURS[backbone]
        jitter = [position + offset for offset in ((index - (len(rows) - 1) / 2) * 0.055 for index in range(len(rows)))]
        overhead_axes.plot(jitter, rows.excess_nll, linestyle='none', marker='o', markersize=3.2,
                           color=colour, alpha=0.85, zorder=3)
        median = rows.excess_nll.median()
        overhead_axes.plot([position - 0.22, position + 0.22], [median, median], color=colour, linewidth=1.6, zorder=4)
        overhead_axes.annotate(f'{median:.03f}', xy=(position + 0.24, median), fontsize=SIZES[1],
                               color=colour, va='center', ha='left')
    overhead_axes.set_xticks(range(len(order)))
    overhead_axes.set_xticklabels([f'{backbone}\n{summary.loc[summary.backbone == backbone, "length"].iloc[0]} aa' for backbone in order])
    overhead_axes.set_ylim(0, None)
    overhead_axes.margins(x=0.22, y=0.12)
    overhead_axes.set_ylabel('overhead (nats per residue)')
    overhead_axes.set_title('b  The controller: excess over exact conditioning', loc='left', fontweight='normal')
    overhead_axes.annotate('lower = better', xy=(0.98, 0.97), xycoords='axes fraction', fontsize=SIZES[1],
                           color='#666666', ha='right', va='top')

    figure.tight_layout(w_pad=1.8)
    os.makedirs(os.path.dirname(arguments.out) or '.', exist_ok=True)
    figure.savefig(arguments.out)
    print(f'wrote {arguments.out}')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
