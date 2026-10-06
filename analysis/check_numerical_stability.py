"""General numerical-stability sweep over results_sqrt_grid/: flags functions whose
negloglike is not trustworthy, across two independent axes.

Grew out of a manual check on `a0 + a1/x**2` (and its reparametrisations
`a1*(a0-1/x**2)`, `a1*(a0+x**(-2))`) which turned out to occupy 3 of the top 5 DL
slots in the comp-8 exhaustive rerun despite being a quadrature artifact: negloglike
115417 / 105.80 / 127.85 at N=120/1000/5000 -- not converging, bouncing around.
Compare `a1/(x*(a0-x))` (the genuine #1): 113.24 / 111.17 / 111.18 -- settles down,
as a real fit should.

Two independent instability checks, same fitted params both times (no refit):
  1. GRID: negloglike at N=120, 1000, 5000 (float64). Flags if N=1000 and N=5000
     disagree beyond tolerance -- N=5000 is esd_fixed.py's own validated default,
     so if the working resolution (1000) doesn't match it, the working-resolution
     number isn't trustworthy.
  2. PRECISION: negloglike at N=1000 in float32 vs float64. Flags if they disagree
     beyond the same tolerance -- catches cases where an extreme fitted parameter
     (e.g. a0 in the millions, as seen in the a0+a1/x**2 family) loses precision
     under JAX's float32 default, which test_all-asmaclap80.py avoids by setting
     jax_enable_x64 but esr_lite.py originally didn't (fixed 2026-08-03).

Tolerance matches audit_quadrature_stability.py's own convention (ABS_TOL=1.0 nat,
REL_TOL=10%) rather than inventing a new one.

For anything flagged, a THIRD pass classifies it rather than just discarding it --
flagging alone conflates two different failure modes that need different treatment.
This pass does a REAL REFIT (esr_lite.refit_one_cluster, per-cluster BFGS warm-started
from the existing params), not just re-evaluation at higher N -- re-evaluating the
stale N=1000-optimized params elsewhere only tells you how bad a wrong guess looks,
not the function's true best-achievable score. Confirmed directly: re-evaluating
a0+a1/x**2's N=1000 params at N=5000/20000/50000 gave nll~127-130 (looked worse than
the single-parameter special case a0/x**2's honest 120.71 -- a nested-model
contradiction, since a0+a1/x**2 can exactly reproduce a0/x**2 by setting the offset to
0, so it can never legitimately score worse). Properly REFITTING a0+a1/x**2 at N=5000
gives nll=119.5 -- consistent, slightly better than the 1-parameter special case, as
nested-model logic requires.
  - INVALID: refit at N=5000 stays inf for every cluster -- a genuine exploit (e.g. a
    pole with no physical relevance), no honest resolution rescues it. Exclude.
  - RESCORED: refit at N=5000 gives a finite total, re-evaluating (not re-fitting
    again) those refit params at N=20000 agrees within tolerance -- the refit result
    is stable, not itself an N=5000-specific artifact -- AND the resulting DL is
    materially different from DL_reported. Use the refit DL, keep the function in the
    ranking at its honest position -- dropping it would throw away a real, computable
    function that might still beat plenty of other candidates.
  - CONFIRMED: same verification as RESCORED (refit + N=20000 agreement), but the
    honest DL turns out to equal DL_reported within tolerance -- the flag was a false
    alarm, not an error in the original number. Distinct from RESCORED so the label
    itself says whether anything needed fixing. Example: pow(Abs(a0),(1/x)) at comp 4
    was flagged via the precision check (float32 gave inf) but the underlying fit was
    always fine -- traced directly to the covariance/Cholesky solve overflowing in
    float32 due to matrix conditioning, not the density function itself (which
    evaluates identically in float32 and float64); refit-in-float64 reproduced
    DL=728.90 exactly. Still worth having checked -- a grid-flagged function and a
    precision-flagged function can each be wrong for reasons the other check can't
    see, so both get the same refit-and-verify treatment regardless of which one
    turns out to be the false alarm.
  - UNRESOLVED: refit succeeds (finite) but disagrees with its own N=20000 check --
    the refit itself may have found a new, resolution-specific local optimum. Needs a
    human look, not an auto-decision either way.

Scope: top --top-k functions by DL per complexity (not the full pool -- with N in
{120,1000,5000} x 2 dtypes, that's 4 evals x 20 clusters per function; exhaustive
would take hours across comp 7-10). Auto-discovers whichever comps have both
final_N.dat and params_compN.pkl under --results-dir. The refit+classify pass only
runs on functions already flagged by the first pass, so its cost scales with the flag
rate, not --top-k -- refitting is much more expensive than evaluating (per-cluster
BFGS, ~4-8s/function vs ~0.2s), which is why it's reserved for the flagged subset.

Meant to run as a post-processing step right after combine_DL_galaxies-asmaclap80.py
(or combine_comp.py / any script writing this same final_N.dat + params_compN.pkl
pair) finishes -- point --results-dir at wherever that landed. Deliberately NOT
integrated into combine_DL_galaxies-asmaclap80.py itself: that script lives outside
this repo (~/OneDrive.../WL/ESR/) and is off-limits for edits here, and it's a pure
aggregator (sums precomputed per-cluster codelen_matches files) with no ESD-evaluation
capability of its own to hook into -- adding the check there would mean adding that
capability, not wiring up an existing signal.

Output: {results-dir}/stability_check_comp{N}.csv (full detail) and
{results-dir}/final_{N}_corrected.dat (same format as final_N.dat -- INVALID rows
dropped, RESCORED/CONFIRMED rows use the verified DL -- so anything downstream that
already reads final_N.dat, e.g. best1000_by_comp.py, can use the corrected file as a
drop-in replacement without changes).
"""
import argparse
import glob
import os
import pickle
import re
import time

import numpy as np

import esr_lite

CLASH = os.path.dirname(os.path.abspath(__file__))
RESULTS = f'{CLASH}/results_sqrt_grid'  # default; overridden by --results-dir in main()

N_LOW, N_MID, N_HIGH = 120, 1000, 5000
N_REFIT = 5000     # refit target for flagged functions -- shown sufficient empirically
N_VHIGH = 20000    # re-evaluate (not re-fit) the refit params here to confirm stability
N_ESCALATE = 50000  # only reached if the N_VHIGH check fails the tight tolerance below --
# a genuine REFIT here (not bare re-evaluation), warm-started from the N_REFIT result
N_VERIFY = 100000   # cheap re-evaluation of the N_ESCALATE refit, to confirm 50000 was
# actually enough rather than just assuming it -- don't trust N_ESCALATE blind
ABS_TOL = 1.0
REL_TOL = 0.10

# Tolerance for the INITIAL flagging pass (nll_1000 vs nll_5000, nll_1000 vs
# nll_1000_f32) -- deliberately loose (10%), matching audit_quadrature_stability.py's
# own convention: this pass only decides "does this function need scrutiny at all,"
# so it's fine to be forgiving of ordinary numerical noise.
#
# Tolerance for CONFIRMING a refit is actually converged, below, is a different and
# much tighter question -- and REL_TOL=0.10 is not tight enough for it. Confirmed
# directly: pow((x*Abs(a0)),a1) at comp 5 showed refit-at-5000=106.92 vs
# confirm-at-20000=108.05, a genuine ~1.1-nat (~1.06%) gap that a 10%-relative check
# waves through as "agreeing" -- but a 3-point chained-refit sequence (5000/20000/
# 40000/100000: 106.92/108.05/108.10/108.12) shows this was real, still-shrinking
# drift, not noise; two points agreeing loosely isn't evidence of convergence, only a
# tight comparison (or a full multi-point sequence) is. CONFIRM_ABS_TOL/REL_TOL below
# are calibrated against genuinely-converged cases seen this session (a0/x**2 agreed
# to <0.0001 nats at every N from 120 to 50000; the honest comp-8 winner
# a1/(x*(a0-x)) agreed to ~0.01 nats between N=1000 and 5000) -- both comfortably
# inside these tighter bounds, while the still-drifting case above is comfortably
# outside them.
CONFIRM_ABS_TOL = 0.3
CONFIRM_REL_TOL = 0.005

FIELDS = ["idx", "function", "DL", "_dup_negloglike", "negloglike", "codelen",
          "ayfeyn", "katz", "a0", "a0_unc", "a1", "a1_unc", "divergence"]


def flagged(a, b, abs_tol=ABS_TOL, rel_tol=REL_TOL):
    """Same tolerance rule as audit_quadrature_stability.py by default (loose,
    for the initial flagging pass). Pass abs_tol/rel_tol=CONFIRM_* for the much
    tighter question "is this refit actually converged" -- see the module-level
    comment above CONFIRM_ABS_TOL for why these need to be different tolerances."""
    if a is None or b is None:
        return True
    import math
    diff = b - a
    if not math.isfinite(diff):
        return not (math.isinf(a) and math.isinf(b))  # both inf/invalid = consistent, not flagged
    return abs(diff) > max(abs_tol, rel_tol * abs(a))


def load_pool(comp):
    """Parse final_{comp}.dat, drop invalid rows, sort by DL ascending."""
    rows = []
    with open(f'{RESULTS}/final_{comp}.dat') as f:
        for line in f:
            parts = line.rstrip('\n').split(';')
            if len(parts) != len(FIELDS):
                continue
            row = dict(zip(FIELDS, parts))
            if row['function'] == '' or row['DL'] == 'nan':
                continue
            try:
                row['DL'] = float(row['DL'])
                row['idx'] = int(row['idx'])
                row['codelen'] = float(row['codelen'])
                row['ayfeyn'] = float(row['ayfeyn'])
            except ValueError:
                continue
            rows.append(row)
    rows.sort(key=lambda r: r['DL'])
    return rows


def _eval_total(eq_numpy, params_per_cluster, names, liks, npts):
    total = 0.0
    for c, name in enumerate(names):
        xvar, yvar, L_factor = liks[name]
        total += esr_lite.eval_nll(eq_numpy, xvar, yvar, L_factor, params_per_cluster[c], npts)
    return total


def classify(eq_numpy, k, params, idx, names, liks, codelen, ayfeyn, nll_1000=None, nll_5000=None):
    """Only called on already-flagged functions. REFITS at a starting resolution
    (not just re-evaluates the stale N=1000 params -- see module docstring for why
    that was wrong), then confirms convergence with a TIGHT tolerance
    (CONFIRM_ABS_TOL/CONFIRM_REL_TOL, not the loose ABS_TOL/REL_TOL the initial
    flagging pass uses -- see the comment above those constants for why two points
    passing a loose check isn't evidence of convergence). If the tight check fails,
    escalate ONCE to a real refit at N_ESCALATE before giving up.

    nll_1000/nll_5000 (optional): the RAW evaluation of the OLD, pre-refit params at
    those two resolutions -- already computed by the caller's initial flagging pass,
    free information otherwise thrown away. If they already agree tightly, this
    function's parameter region is probably not resolution-sensitive, so start the
    refit at N_REFIT as usual. If they disagree badly (e.g. a0/x+a1: 210 -> 1833),
    that's a strong hint N_REFIT won't be enough either -- skip straight to starting
    the refit at N_ESCALATE instead of wasting a refit-at-N_REFIT attempt that's
    likely to fail the tight check anyway. This is a heuristic, not a guarantee (the
    refit-optimal params can sit in a different, more- or less-sensitive region than
    the old params) -- it only decides where to START, the same tight-check-then-
    escalate logic below still applies regardless.

    Returns (classification, corrected_DL_or_None, nll_refit, nll_high_on_refit_params,
    refit_params_per_cluster) -- the last two are None unless classification is
    RESCORED (the label gets relabeled to CONFIRMED by the caller if the corrected DL
    turns out to equal DL_reported within the loose tolerance).
    """
    start_n = N_REFIT
    if nll_1000 is not None and nll_5000 is not None and flagged(nll_1000, nll_5000):
        start_n = N_ESCALATE

    nll_refit_total = 0.0
    refit_params_per_cluster = []
    for c, name in enumerate(names):
        xvar, yvar, L_factor = liks[name]
        p0 = [float(params[f'a{j}'][idx, c]) for j in range(k)]
        nll_c, p_refit = esr_lite.refit_one_cluster(eq_numpy, xvar, yvar, L_factor, p0, k, start_n)
        nll_refit_total += nll_c
        refit_params_per_cluster.append(p_refit)

    if nll_refit_total == float('inf'):
        return 'INVALID', None, nll_refit_total, None, None

    # Confirm resolution must be HIGHER than wherever we actually started the refit
    # (N_VHIGH=20000 would be a downgrade, not a confirmation, if start_n was already
    # N_ESCALATE=50000 because the cheap pre-check flagged the old params as unstable).
    confirm_n = N_VHIGH if start_n < N_VHIGH else N_VERIFY
    nll_confirm_total = _eval_total(eq_numpy, refit_params_per_cluster, names, liks, confirm_n)

    if not flagged(nll_refit_total, nll_confirm_total, CONFIRM_ABS_TOL, CONFIRM_REL_TOL):
        corrected_DL = nll_confirm_total + codelen + ayfeyn
        return 'RESCORED', corrected_DL, nll_refit_total, nll_confirm_total, refit_params_per_cluster

    if start_n >= N_ESCALATE:
        # Already started at the high tier and even N_VERIFY disagrees -- no further
        # predefined escalation. Genuinely hard case, not a labeling failure.
        return 'UNRESOLVED', None, nll_refit_total, nll_confirm_total, None

    # Tight check failed at N_VHIGH -- escalate to a REAL refit at N_ESCALATE (not
    # bare re-evaluation: a bare re-eval only tells you how the N_REFIT-tuned params
    # happen to score elsewhere, not whether N_ESCALATE has its own, possibly better,
    # optimum -- the same "re-evaluate isn't refit" mistake this whole module exists
    # to avoid, just one level up). Warm-started from the N_REFIT result.
    nll_escalate_total = 0.0
    escalate_params_per_cluster = []
    for c, name in enumerate(names):
        xvar, yvar, L_factor = liks[name]
        nll_c, p_esc = esr_lite.refit_one_cluster(
            eq_numpy, xvar, yvar, L_factor, refit_params_per_cluster[c], k, N_ESCALATE)
        nll_escalate_total += nll_c
        escalate_params_per_cluster.append(p_esc)

    if nll_escalate_total == float('inf'):
        return 'UNRESOLVED', None, nll_refit_total, nll_escalate_total, None

    # Don't trust N_ESCALATE blind either -- confirm IT has actually converged too,
    # via a cheap re-evaluation at N_VERIFY (bare eval is fine here: this is a check
    # on the escalation refit's own stability, not a new optimum-finding step).
    nll_verify_total = _eval_total(eq_numpy, escalate_params_per_cluster, names, liks, N_VERIFY)
    if not flagged(nll_escalate_total, nll_verify_total, CONFIRM_ABS_TOL, CONFIRM_REL_TOL):
        corrected_DL = nll_verify_total + codelen + ayfeyn
        return 'RESCORED', corrected_DL, nll_refit_total, nll_verify_total, escalate_params_per_cluster

    # Still disagreeing even after refit-at-50000 + verify-at-100000: this is a
    # genuinely hard case (per esd_fixed.py's own documented residual-risk caveat --
    # some features can evade any finite grid), not a labeling failure. Don't
    # auto-decide; flag for a human look.
    return 'UNRESOLVED', None, nll_refit_total, nll_verify_total, None


def discover_comps():
    comps = []
    for path in sorted(glob.glob(f'{RESULTS}/final_*.dat')):
        m = re.search(r'final_(\d+)\.dat$', path)
        if m and os.path.exists(f'{RESULTS}/params_comp{m.group(1)}.pkl'):
            comps.append(int(m.group(1)))
    return comps


def main():
    global RESULTS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--top-k', type=int, default=200,
                         help='functions checked per complexity, ranked by DL')
    parser.add_argument('--results-dir', default=RESULTS,
                         help='directory with final_N.dat + params_compN.pkl to check '
                              '(e.g. wherever combine_DL_galaxies-asmaclap80.py just wrote '
                              'its output). Defaults to results_sqrt_grid/.')
    parser.add_argument('--comps', default=None,
                         help='comma-separated complexities to check (e.g. "9" or "9,10"). '
                              'Default: all discovered comps -- use this to avoid redoing '
                              'ones already checked.')
    args = parser.parse_args()

    RESULTS = os.path.abspath(args.results_dir)

    names = open(f'{CLASH}/cluster_names.txt').read().splitlines()
    liks = {name: esr_lite.load_likelihood(name) for name in names}

    comps = discover_comps()
    if args.comps is not None:
        requested = {int(c) for c in args.comps.split(',')}
        comps = [c for c in comps if c in requested]
    print(f'Complexities found (final_N.dat + params_compN.pkl both present): {comps}')

    for comp in comps:
        pool = load_pool(comp)
        top = pool[:args.top_k]
        print(f'\ncomp {comp}: {len(pool)} valid functions, checking top {len(top)} by DL')

        with open(f'{RESULTS}/params_comp{comp}.pkl', 'rb') as f:
            params = pickle.load(f)

        t0 = time.time()
        results = []
        for row in top:
            idx, fn, DL_reported, codelen, ayfeyn = (
                row['idx'], row['function'], row['DL'], row['codelen'], row['ayfeyn'])
            eq_numpy, k = esr_lite.get_eq_numpy(fn)

            nll = {}
            for npts in (N_LOW, N_MID, N_HIGH):
                total = 0.0
                for c, name in enumerate(names):
                    xvar, yvar, L_factor = liks[name]
                    p = [float(params[f'a{j}'][idx, c]) for j in range(k)]
                    total += esr_lite.eval_nll(eq_numpy, xvar, yvar, L_factor, p, npts)
                nll[npts] = total

            nll_f32 = 0.0
            for c, name in enumerate(names):
                xvar, yvar, L_factor = liks[name]
                p = [float(params[f'a{j}'][idx, c]) for j in range(k)]
                nll_f32 += esr_lite.eval_nll(eq_numpy, xvar, yvar, L_factor, p, N_MID,
                                              dtype=esr_lite.jnp.float32)

            grid_flag = flagged(nll[N_MID], nll[N_HIGH])
            precision_flag = flagged(nll[N_MID], nll_f32)
            any_flag = grid_flag or precision_flag

            classification, corrected_DL, nll_refit, nll_confirm, refit_p = (
                None, None, None, None, None)
            if any_flag:
                classification, corrected_DL, nll_refit, nll_confirm, refit_p = classify(
                    eq_numpy, k, params, idx, names, liks, codelen, ayfeyn,
                    nll_1000=nll[N_MID], nll_5000=nll[N_HIGH])
                # RESCORED means "verified via refit, here's the honest DL" -- but that
                # honest DL sometimes equals the original (e.g. a precision_flag turns
                # out to be a false alarm the float64 refit reproduces exactly, as with
                # pow(Abs(a0),(1/x)): flagged because float32 overflowed in the
                # covariance solve, not because the fit was wrong). Relabel those
                # CONFIRMED so the label itself says whether anything needed fixing,
                # instead of RESCORED covering both "changed" and "verified unchanged."
                if classification == 'RESCORED' and not flagged(DL_reported, corrected_DL):
                    classification = 'CONFIRMED'

            results.append({
                'idx': idx, 'fn': fn, 'DL_reported': DL_reported,
                'nll_120': nll[N_LOW], 'nll_1000': nll[N_MID], 'nll_5000': nll[N_HIGH],
                'nll_1000_f32': nll_f32,
                'grid_flag': grid_flag, 'precision_flag': precision_flag, 'any_flag': any_flag,
                'classification': classification, 'corrected_DL': corrected_DL,
                'nll_refit_5000': nll_refit, 'nll_confirm_20000': nll_confirm,
                'refit_params': refit_p, 'codelen': codelen, 'ayfeyn': ayfeyn,
            })

        dt = time.time() - t0
        n_grid = sum(1 for r in results if r['grid_flag'])
        n_prec = sum(1 for r in results if r['precision_flag'])
        n_any = sum(1 for r in results if r['any_flag'])
        n_invalid = sum(1 for r in results if r['classification'] == 'INVALID')
        n_rescored = sum(1 for r in results if r['classification'] == 'RESCORED')
        n_confirmed = sum(1 for r in results if r['classification'] == 'CONFIRMED')
        n_unresolved = sum(1 for r in results if r['classification'] == 'UNRESOLVED')
        print(f'  {n_grid} grid-unstable, {n_prec} precision-unstable, '
              f'{n_any} flagged total ({dt:.1f}s)')
        print(f'  of those: {n_invalid} INVALID (exclude), {n_rescored} RESCORED '
              f'(value changed, use corrected DL), {n_confirmed} CONFIRMED '
              f'(verified, DL unchanged), {n_unresolved} UNRESOLVED (needs a manual look)')

        # Ranking as it actually stands among the checked top-k once INVALID is
        # excluded and RESCORED/CONFIRMED use the corrected DL -- this is the
        # trustworthy view, not DL_reported. (Functions outside top-k weren't
        # checked at all, so this is "best of what we looked at," not "best overall.")
        effective = [
            (r['corrected_DL'] if r['classification'] in ('RESCORED', 'CONFIRMED') else r['DL_reported'],
             r['fn'], r['classification'])
            for r in results if r['classification'] != 'INVALID'
        ]
        effective.sort(key=lambda t: t[0])
        print(f'  effective #1 after correction: DL={effective[0][0]:.2f}  '
              f'{"[" + effective[0][2] + "]" if effective[0][2] else ""}  {effective[0][1]}')

        n_flagged_shown = 0
        for r in sorted(results, key=lambda r: r['DL_reported']):
            if r['any_flag'] and n_flagged_shown < 10:
                reasons = []
                if r['grid_flag']:
                    reasons.append('grid')
                if r['precision_flag']:
                    reasons.append('precision')
                corr = f" -> corrected DL={r['corrected_DL']:.2f}" if r['corrected_DL'] else ""
                print(f"    DL={r['DL_reported']:8.2f}  [{','.join(reasons):18s}]  "
                      f"{r['classification']:11s}{corr}  {r['fn']}")
                n_flagged_shown += 1

        def fmt(v):
            return f'{v:.4f}' if v is not None else ''

        out_path = f'{RESULTS}/stability_check_comp{comp}.csv'
        with open(out_path, 'w') as f:
            f.write('idx;fn;DL_reported;nll_120;nll_1000;nll_5000;nll_1000_f32;'
                    'nll_refit_5000;nll_confirm_20000;grid_flag;precision_flag;any_flag;'
                    'classification;corrected_DL\n')
            for r in results:
                f.write(f"{r['idx']};{r['fn']};{r['DL_reported']:.4f};"
                        f"{r['nll_120']:.4f};{r['nll_1000']:.4f};{r['nll_5000']:.4f};"
                        f"{r['nll_1000_f32']:.4f};{fmt(r['nll_refit_5000'])};"
                        f"{fmt(r['nll_confirm_20000'])};"
                        f"{r['grid_flag']};{r['precision_flag']};{r['any_flag']};"
                        f"{r['classification'] or ''};{fmt(r['corrected_DL'])}\n")
        print(f'  -> {out_path}')

        # final_{comp}_corrected.dat -- same format as final_comp.dat, so anything
        # downstream that reads final_N.dat can use this as a drop-in replacement.
        # INVALID rows dropped; RESCORED/CONFIRMED rows get the refit DL/negloglike/
        # a0/a1 (CONFIRMED's just happens to equal the original within tolerance --
        # still the verified value, not blindly copied); everything else (unflagged,
        # or outside top-k and never checked) unchanged.
        by_idx = {r['idx']: r for r in results}
        invalid_idx = {r['idx'] for r in results if r['classification'] == 'INVALID'}
        corrected_path = f'{RESULTS}/final_{comp}_corrected.dat'
        with open(corrected_path, 'w') as f:
            for row in pool:
                if row['idx'] in invalid_idx:
                    continue
                r = by_idx.get(row['idx'])
                if r is not None and r['classification'] in ('RESCORED', 'CONFIRMED'):
                    new_negloglike = r['nll_confirm_20000']
                    new_DL = r['corrected_DL']
                    # plain numpy, not statistics.pstdev -- pstdev uses exact Fraction
                    # arithmetic internally and raises AttributeError on any inf/nan in
                    # the input (confirmed: crashed mid-sweep on comp 6). numpy just lets
                    # inf/nan propagate into the mean/std, which is the honest outcome if
                    # a cluster's refit genuinely landed somewhere non-finite -- visible in
                    # the output rather than silently dropped or fatal.
                    p = r['refit_params']  # list of per-cluster [a0,(a1)] lists
                    a0_vals = [pc[0] for pc in p] if p and len(p[0]) > 0 else [0.0]
                    a1_vals = [pc[1] for pc in p] if p and len(p[0]) > 1 else [0.0]
                    a0_mean, a0_unc = float(np.mean(a0_vals)), float(np.std(a0_vals))
                    a1_mean, a1_unc = float(np.mean(a1_vals)), float(np.std(a1_vals))
                    out_fields = [
                        str(row['idx']), row['function'], f'{new_DL:.6f}',
                        f'{new_negloglike:.6f}', f'{new_negloglike:.6f}',
                        f"{row['codelen']:.6f}", f"{row['ayfeyn']:.6f}", row['katz'],
                        f'{a0_mean:.6f}', f'{a0_unc:.6f}', f'{a1_mean:.6f}', f'{a1_unc:.6f}',
                        r['classification'],
                    ]
                else:
                    out_fields = [row[field] if field not in ('DL', 'idx', 'codelen', 'ayfeyn')
                                  else str(row[field]) for field in FIELDS]
                    if r is not None and r['classification'] == 'UNRESOLVED':
                        out_fields[-1] = 'UNRESOLVED'  # divergence column -- flag, don't silently pass through
                f.write(';'.join(out_fields) + '\n')
        print(f'  -> {corrected_path} ({len(invalid_idx)} INVALID rows dropped)')


if __name__ == '__main__':
    main()
