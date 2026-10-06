"""Cheap triage for the old-pool-rank-1000-3000 tail probe (build_tail_probe_candidates.py).

Two stages, both local, no cluster/MPI:

Stage 1 (cheap re-score, no refit): reuse each candidate's EXISTING per-cluster params
from results/params_comp<file_comp>.pkl (the ORIGINAL raw fit -- optimised for the old,
pre-sqrt-grid-fix objective) and re-score under the corrected likelihood at
num_points=1000, matching results_sqrt_grid/'s operating point. This is a LOWER-BOUND
triage, not a fair comparison: a function that was quietly exploiting the old quadrature
bug can score inf here even though a proper refit of the same functional form might fit
reasonably. So:
  - Candidates that already score worse than `threshold` (the best DL found so far among
    the properly-refit results_sqrt_grid/ shortlist, or the old top-1000 cutoff if no comp
    has been refit yet) are dropped -- a refit can only look BETTER than an honest
    re-score, so this side is safe to cut.
  - The rest (competitive, or inf/invalid -- ambiguous, could be a demoted exploiter) are
    candidates for Stage 2.

Stage 2 (single-start local refit, --refit): for a bounded set of Stage-1 survivors
(top --refit-top-k by cheap DL, plus --refit-inf-sample of the inf ones by original DL),
run ONE per-cluster BFGS refit (jax.grad, warm-started from the old params) -- not the
full multi-start basin-hopping test_all-asmaclap80.py uses, but enough to tell whether an
inf/mediocre cheap score was really the function's fault or just its stale old params.
Anything that still looks competitive after this is flagged NEEDS_CLUSTER_REFIT --
genuinely fitting it fairly needs the full multi-start pipeline on the cluster, which is
out of scope here.

Output: tail_probe_1000_3000/triage_comp<N>.csv
"""
import argparse
import glob
import os

import numpy as np

import esr_lite

CLASH = os.path.dirname(os.path.abspath(__file__))
TAIL_DIR = f'{CLASH}/tail_probe_1000_3000'
RESULTS_RAW = f'{CLASH}/results'
NPTS = 1000


def get_threshold():
    """Best DL found so far among the refit results_sqrt_grid/ shortlist, else the
    old top-1000 cutoff (so triage is meaningful even before any comp is refit)."""
    best = None
    for path in glob.glob(f'{CLASH}/results_sqrt_grid/resolution_check_comp*.csv'):
        with open(path) as f:
            next(f)  # header
            for line in f:
                parts = line.rstrip('\n').split(';')
                dl = float(parts[6])  # DL_1000
                if np.isfinite(dl) and (best is None or dl < best):
                    best = dl
    if best is not None:
        return best, 'best DL among refit results_sqrt_grid/ comps'
    cutoff_path = f'{TAIL_DIR}/top1000_cutoff_DL.txt'
    with open(cutoff_path) as f:
        return float(f.read().strip()), 'old top-1000 cutoff DL (no comp refit yet)'


def load_candidates():
    """comp -> list of dict(overall_rank, idx, file_comp, fn, DL_orig, nll_orig, codelen)."""
    by_comp = {}
    for path in sorted(glob.glob(f'{TAIL_DIR}/compl_*/functions_*.txt')):
        comp = int(os.path.basename(path).replace('functions_', '').replace('.txt', ''))
        rows = []
        with open(path) as f:
            for line in f:
                parts = line.rstrip('\n').split(';')
                rows.append({
                    'overall_rank': int(parts[0]), 'idx': int(parts[1]),
                    'file_comp': int(parts[2]), 'fn': parts[3],
                    'DL_orig': float(parts[4]), 'nll_orig': float(parts[5]),
                    'codelen': float(parts[6]),
                })
        by_comp[comp] = rows
    return by_comp


def cheap_rescore(row, names, liks, params_cache):
    file_comp = row['file_comp']
    if file_comp not in params_cache:
        import pickle
        with open(f'{RESULTS_RAW}/params_comp{file_comp}.pkl', 'rb') as f:
            params_cache[file_comp] = pickle.load(f)
    params = params_cache[file_comp]

    eq_numpy, k = esr_lite.get_eq_numpy(row['fn'])
    idx = row['idx']
    total_nll = 0.0
    per_cluster = []
    for c, name in enumerate(names):
        xvar, yvar, L_factor = liks[name]
        p = [float(params[f'a{j}'][idx, c]) for j in range(k)]
        nll = esr_lite.eval_nll(eq_numpy, xvar, yvar, L_factor, p, NPTS)
        total_nll += nll
        per_cluster.append(p)
    return total_nll, per_cluster, k, eq_numpy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refit', action='store_true',
                         help='Also run Stage 2 (single-start local refit) on survivors')
    parser.add_argument('--refit-top-k', type=int, default=30)
    parser.add_argument('--refit-inf-sample', type=int, default=30)
    args = parser.parse_args()

    names = open(f'{CLASH}/cluster_names.txt').read().splitlines()
    liks = {name: esr_lite.load_likelihood(name) for name in names}

    threshold, threshold_src = get_threshold()
    print(f'Triage threshold: DL < {threshold:.4f} ({threshold_src})\n')

    by_comp = load_candidates()
    params_cache = {}

    for comp, rows in by_comp.items():
        scored = []
        for row in rows:
            total_nll, per_cluster, k, eq_numpy = cheap_rescore(row, names, liks, params_cache)
            DL_cheap = total_nll + row['codelen']
            scored.append({**row, 'DL_cheap': DL_cheap, 'nll_cheap': total_nll,
                           'per_cluster_params': per_cluster, 'k': k, 'eq_numpy': eq_numpy})

        finite = [r for r in scored if np.isfinite(r['DL_cheap'])]
        infinite = [r for r in scored if not np.isfinite(r['DL_cheap'])]
        finite.sort(key=lambda r: r['DL_cheap'])
        infinite.sort(key=lambda r: r['DL_orig'])

        beats = [r for r in finite if r['DL_cheap'] < threshold]
        refit_set = finite[:args.refit_top_k] + infinite[:args.refit_inf_sample]
        refit_ids = {r['idx'] for r in refit_set}

        print(f"comp {comp}: {len(scored)} candidates, {len(infinite)} inf/invalid "
              f"under cheap re-score, {len(beats)} ALREADY beat threshold without refit")
        if beats:
            print(f"  *** {len(beats)} candidate(s) already beat the threshold using "
                  f"un-refit old params -- strong signal, needs cluster refit ***")
            for r in beats[:5]:
                print(f"    DL_cheap={r['DL_cheap']:.2f}  {r['fn']}")

        results = []
        for r in scored:
            verdict = 'below_threshold_dropped'
            refit_nll = refit_dl = None
            if r['DL_cheap'] < threshold:
                verdict = 'BEATS_THRESHOLD_needs_cluster_refit'
            elif r['idx'] in refit_ids:
                verdict = 'refit_candidate'

            if args.refit and r['idx'] in refit_ids:
                total_refit_nll = 0.0
                for c, name in enumerate(names):
                    xvar, yvar, L_factor = liks[name]
                    p0 = r['per_cluster_params'][c]
                    nll_c, _ = esr_lite.refit_one_cluster(r['eq_numpy'], xvar, yvar, L_factor, p0, r['k'], NPTS)
                    total_refit_nll += nll_c
                refit_nll = total_refit_nll
                refit_dl = refit_nll + r['codelen']
                if refit_dl < threshold:
                    verdict = 'REFIT_BEATS_THRESHOLD_needs_cluster_refit'
                elif verdict == 'refit_candidate':
                    verdict = 'refit_checked_still_worse'

            results.append({**r, 'refit_nll': refit_nll, 'refit_DL': refit_dl, 'verdict': verdict})

        out_path = f'{TAIL_DIR}/triage_comp{comp}.csv'
        with open(out_path, 'w') as f:
            f.write('overall_rank;idx;fn;DL_orig;DL_cheap;refit_DL;verdict\n')
            for r in results:
                refit_dl_str = f"{r['refit_DL']:.4f}" if r['refit_DL'] is not None else ''
                f.write(f"{r['overall_rank']};{r['idx']};{r['fn']};{r['DL_orig']:.4f};"
                        f"{r['DL_cheap']:.4f};{refit_dl_str};{r['verdict']}\n")
        print(f"  -> {out_path}\n")


if __name__ == '__main__':
    main()
