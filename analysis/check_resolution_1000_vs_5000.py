"""Resolution convergence check for results_sqrt_grid/: num_points=1000 vs 5000.

For every complexity with a finished results_sqrt_grid/params_comp{N}_1k_funcs.pkl,
re-evaluates each function's total negloglike (summed over all 20 clusters) at
num_points=1000 (what was actually used, reduced from the cluster-cost-driven
default) and num_points=5000 (esd_fixed.py's own DEFAULT_NUM_POINTS, validated in
notes.md's convergence study). Same fitted params both times -- no refit, mirrors
audit_quadrature_stability.py's low/high-N pattern.

The question that matters operationally isn't "does negloglike drift a little" --
it's whether the two resolutions would make you pick a DIFFERENT best function.
So this reports both the per-function drift AND whether the within-comp ranking
(by DL) is invariant between the two resolutions.

Outputs, per comp:
  results_sqrt_grid/resolution_check_comp{N}.csv
"""
import glob
import math
import os
import pickle
import re

import esr_lite

CLASH = os.path.dirname(os.path.abspath(__file__))
RESULTS = f'{CLASH}/results_sqrt_grid'

NPTS_LOW = 1000
NPTS_HIGH = 5000
ABS_TOL = 1.0    # nats, same tolerance audit_quadrature_stability.py uses
REL_TOL = 0.10


def load_pretty_table(path):
    """Parse the '| # | Function | L(D) | -logL | Codelen | ... |' table."""
    rows = []
    with open(path) as f:
        for line in f:
            if not line.startswith('|'):
                continue
            cols = [c.strip() for c in line.strip().strip('|').split('|')]
            if not cols or cols[0] == '#' or not cols[0].lstrip('-').isdigit():
                continue
            rows.append({
                'idx': int(cols[0]),
                'fn': cols[1],
                'DL_reported': float(cols[2]),
                'nll_reported': float(cols[3]),
                'codelen': float(cols[4]),
            })
    return rows


def discover_comps():
    comps = []
    for path in sorted(glob.glob(f'{RESULTS}/params_comp*_1k_funcs.pkl')):
        m = re.search(r'params_comp(\d+)_1k_funcs\.pkl', path)
        if m:
            comps.append(int(m.group(1)))
    return comps


def main():
    names = open(f'{CLASH}/cluster_names.txt').read().splitlines()
    liks = {name: esr_lite.load_likelihood(name) for name in names}

    comps = discover_comps()
    if not comps:
        print('No results_sqrt_grid/params_comp*_1k_funcs.pkl found.')
        return

    for comp in comps:
        pretty_path = f'{RESULTS}/pretty_all_clusters_comp{comp}_1k_funcs.txt'
        pkl_path = f'{RESULTS}/params_comp{comp}_1k_funcs.pkl'
        if not os.path.exists(pretty_path):
            print(f'comp={comp}: missing {pretty_path}, skipping')
            continue

        rows = load_pretty_table(pretty_path)
        with open(pkl_path, 'rb') as f:
            params = pickle.load(f)

        results = []
        for row in rows:
            idx, fn, codelen = row['idx'], row['fn'], row['codelen']
            eq_numpy, k = esr_lite.get_eq_numpy(fn)

            nll_low, nll_high = 0.0, 0.0
            for c, name in enumerate(names):
                xvar, yvar, L_factor = liks[name]
                p = [params[f'a{j}'][idx, c] for j in range(k)]
                nll_low += esr_lite.eval_nll(eq_numpy, xvar, yvar, L_factor, p, NPTS_LOW)
                nll_high += esr_lite.eval_nll(eq_numpy, xvar, yvar, L_factor, p, NPTS_HIGH)

            diff = nll_high - nll_low
            if not math.isfinite(diff):
                flagged = True
            else:
                flagged = abs(diff) > max(ABS_TOL, REL_TOL * abs(nll_low))
            results.append({
                'idx': idx, 'fn': fn,
                'nll_reported': row['nll_reported'],
                'nll_1000': nll_low, 'nll_5000': nll_high, 'diff': diff,
                'DL_1000': nll_low + codelen, 'DL_5000': nll_high + codelen,
                'flagged': flagged,
            })

        results.sort(key=lambda r: r['DL_1000'])
        rank_1000 = {r['idx']: i for i, r in enumerate(results)}
        results_by_5000 = sorted(results, key=lambda r: r['DL_5000'])
        rank_5000 = {r['idx']: i for i, r in enumerate(results_by_5000)}

        n_flagged = sum(1 for r in results if r['flagged'])
        rank_changed = [r['idx'] for r in results if rank_1000[r['idx']] != rank_5000[r['idx']]]
        top1_1000 = results[0]['fn']
        top1_5000 = results_by_5000[0]['fn']

        print(f"comp={comp}: {len(results)} functions, {n_flagged} flagged "
              f"(|diff| > max({ABS_TOL}, {REL_TOL}*|nll@1000|)), "
              f"{len(rank_changed)}/{len(results)} change rank between N=1000 and N=5000")
        print(f"  best @ N=1000: {top1_1000}")
        print(f"  best @ N=5000: {top1_5000}"
              + ('  <-- DIFFERENT WINNER' if top1_1000 != top1_5000 else '  (same)'))

        out_path = f'{RESULTS}/resolution_check_comp{comp}.csv'
        with open(out_path, 'w') as f:
            f.write('idx;fn;nll_reported;nll_1000;nll_5000;diff;DL_1000;DL_5000;'
                    'rank_1000;rank_5000;flagged\n')
            for r in results:
                f.write(f"{r['idx']};{r['fn']};{r['nll_reported']:.4f};"
                        f"{r['nll_1000']:.4f};{r['nll_5000']:.4f};{r['diff']:.4f};"
                        f"{r['DL_1000']:.4f};{r['DL_5000']:.4f};"
                        f"{rank_1000[r['idx']]};{rank_5000[r['idx']]};{r['flagged']}\n")
        print(f"  -> {out_path}")


if __name__ == '__main__':
    main()
