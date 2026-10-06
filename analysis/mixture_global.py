#!/usr/bin/env python3
"""
Globally-optimal profile -> cluster assignment for CLASH mixture models.

Rebuilds the universal-vs-mixture profile->cluster analysis (the paper section
5.4 comparison) for CLASH, minimising the TOTAL description length (fit +
structure + assignment) rather than choosing greedily. The old scripts
(all_two_models_2.py for M=2, different_models_2.py for the per-cluster row) let
each cluster pick argmin(negloglike + codelen) and bolted on the assignment cost
AFTER the split was frozen -- a local choice, because L_assign depends only on
the counts, so a near-tie cluster can be sent the wrong way.

Emits these models under one consistent cost (written to model_comparison.txt):
  - Universal (M=1) : every cluster on one shared function; L_assign = 0.
  - Two profiles (M=2) : EXACT. Sort clusters by the fit-gap delta_i =
        L_a(i) - L_b(i) and sweep the threshold. L_assign is constant on each of
        the N+1 segments, so each split is O(1) from prefix sums -> exact global
        optimum in O(N log N) per pair. (Greedy = the single threshold at
        delta=0, so global is provably never worse.)
  - Per-cluster : coordinate descent ("hard-EM") over the full candidate pool,
        minimising the total DL; multi-restart to de-risk local minima. The
        globally-optimised counterpart of different_models_2.py's greedy choice.

Assignment cost uses the entropy/Shannon form from paper Eq. (34):
    L_assign = - sum_m n_m * log(n_m / N)
consistently for every model (--assignment enumerative reproduces the old
per-cluster script's Option A). Old scripts are left untouched.

Fits are read from the best-funcs run (single source; see CONFIGURATION).

This is HPC code: fit files live on Glamdring. Run it there.
`python mixture_global.py --self-test` checks the maths locally (no data files).
"""

import argparse
import itertools
import math
from pathlib import Path

import numpy as np

# ==========================================
# CONFIGURATION  (CLASH)
# ==========================================
#
# SINGLE SOURCE: the best-funcs run
# ---------------------------------
# We only rank the top-20 functions, and those all live in the best-funcs
# library (the merged/snapped re-run, more optimiser iterations -> better
# convergence). So there is one library + one fit run; no per-function
# best-then-main fallback. (If you ever widen the pool to functions only in
# core_maths, add a "main" entry to SOURCES and put it in SOURCE_ORDER.)
#
# Index lookup: match in all_equations, NOT unique_equations. unique_equations
# stores one canonical variant per class, which need not be the variant that a
# ranking-table string (or the actual fitted row) uses. all_equations row i lines
# up 1:1 with codelen_matches_comp{c}.dat row i, so:
#   all_equations_{c}.txt       row i  = the exact variant ESR generated
#   codelen_matches_comp{c}.dat row i  = (negloglike, codelen) for that row
# A function's fit = min over all rows i whose canonical form == the function's.

COMBINE_FILE = "../esr/fitting/output/combining_clusters/combine_all_comp_all_clustersbest_funcs_change_snapping_merged.dat"
CLUSTERS_LIST = "../esr/data/cluster_names.txt"
OUTPUT_DIR = Path("esr/fitting/output/combining_clusters")

FIT_ROOT_DIR = "../esr/fitting/output"
LIB_BASE = "../esr/function_library"

# Per-source: (function library root, cluster-folder suffix).
SOURCES = {
    "best": {
        "lib": f"{LIB_BASE}/best_funcs_change_snapping",
        "suffix": "_best_funcs_change_snapping_2",
    },
}
SOURCE_ORDER = ["best"]   # single source; add "main" here to re-enable fallback

# Folder suffix of the best-funcs RE-RUN (better convergence / snapping). Used by
# the full-library per-cluster model, which pools the base run (output_WL_<c>)
# with this re-run and prefers the re-run's fit where it exists / is better.
BEST_RERUN_SUFFIX = SOURCES["best"]["suffix"]

TOP_N = 20


# ==========================================
# ASSIGNMENT COST
# ==========================================

def assignment_cost(counts, kind="entropy"):
    """Cost of encoding which cluster uses which profile.

    counts : iterable of n_m (only the used functions; zeros are ignored).
    entropy    -> -sum n_m log(n_m/N)         (paper Eq. 34)
    enumerative-> log N! - sum log n_m!        (old different_models_2.py Option A)
    Returns 0 when only one function is used (single-profile => no label cost).
    """
    counts = [c for c in counts if c > 0]
    N = sum(counts)
    if N == 0 or len(counts) <= 1:
        return 0.0
    if kind == "entropy":
        return -sum(c * math.log(c / N) for c in counts)
    if kind == "enumerative":
        return math.lgamma(N + 1) - sum(math.lgamma(c + 1) for c in counts)
    raise ValueError(f"unknown assignment kind: {kind}")


# ==========================================
# CORE SOLVERS  (operate on an L-matrix; no I/O)
# ==========================================

def greedy_pair(La, Lb, lib_a, lib_b, assignment="entropy"):
    """Old method: each cluster -> argmin(L_a, L_b), assignment added after."""
    to_a = La <= Lb
    n_a = int(to_a.sum())
    n_b = int((~to_a).sum())
    fit = float(La[to_a].sum() + Lb[~to_a].sum())
    lib = (lib_a if n_a > 0 else 0.0) + (lib_b if n_b > 0 else 0.0)
    assign = assignment_cost([n_a, n_b], assignment)
    return {"n_a": n_a, "n_b": n_b, "fit": fit, "lib": lib,
            "assign": assign, "total": fit + lib + assign}


def optimal_pair(La, Lb, lib_a, lib_b, assignment="entropy", require_both=True):
    """Exact global optimum for a fixed pair via threshold sweep.

    Cluster i is assigned to A iff delta_i = La(i) - Lb(i) is among the k
    smallest gaps, for some cut k in {0, ..., N}. We sort by delta once and
    evaluate all N+1 cuts using prefix sums. inf entries (no valid fit on one
    side) are handled: a cluster with La=inf can never be in A, etc.

    require_both=True forces a genuine TWO-function split (n_a >= 1 and n_b >= 1),
    so a pair cannot collapse to a single function (that is the M=1 model's job).
    Returns None if no valid two-function split exists (e.g. every cluster can
    only be fit by one of the two functions).
    """
    La = np.asarray(La, dtype=float)
    Lb = np.asarray(Lb, dtype=float)
    N = La.shape[0]

    # Clusters that can only live on one side (the other fit is inf/invalid).
    only_b = np.isinf(La) & np.isfinite(Lb)   # must go to B
    only_a = np.isinf(Lb) & np.isfinite(La)   # must go to A
    both = np.isfinite(La) & np.isfinite(Lb)
    dead = np.isinf(La) & np.isinf(Lb)        # no fit either way -> drop
    if dead.any():
        keep = ~dead
        La, Lb, only_b, only_a, both = La[keep], Lb[keep], only_b[keep], only_a[keep], both[keep]
        N = La.shape[0]

    # Fixed contributions from the forced clusters.
    fixed_fit = float(La[only_a].sum() + Lb[only_b].sum())
    fixed_na = int(only_a.sum())
    fixed_nb = int(only_b.sum())

    # Free clusters: sort by gap; smaller gap => cheaper on A.
    idx = np.where(both)[0]
    gaps = (La - Lb)[idx]
    order = np.argsort(gaps, kind="mergesort")
    La_s = La[idx][order]
    Lb_s = Lb[idx][order]
    M = len(idx)

    # Prefix sums so any cut is O(1). cut = k free clusters go to A (the k
    # smallest gaps); the rest go to B.
    csum_La = np.concatenate([[0.0], np.cumsum(La_s)])
    total_Lb = float(Lb_s.sum())
    csum_Lb = np.concatenate([[0.0], np.cumsum(Lb_s)])

    best = None
    for k in range(M + 1):
        n_a = fixed_na + k
        n_b = fixed_nb + (M - k)
        if require_both and (n_a == 0 or n_b == 0):
            continue                      # reject degenerate single-function splits
        fit_free = csum_La[k] + (total_Lb - csum_Lb[k])
        lib = (lib_a if n_a > 0 else 0.0) + (lib_b if n_b > 0 else 0.0)
        assign = assignment_cost([n_a, n_b], assignment)
        total = fixed_fit + fit_free + lib + assign
        if best is None or total < best["total"]:
            best = {"n_a": n_a, "n_b": n_b, "fit": fixed_fit + fit_free,
                    "lib": lib, "assign": assign, "total": total, "cut_k": k}
    return best      # None if require_both and no valid two-function split exists


def coordinate_descent(L, lib, assignment="entropy", init=None,
                       max_iter=200, rng=None):
    """Coordinate descent (hard-EM) for M>=3 over a candidate function pool.

    L    : (N_clusters, N_funcs) matrix of negloglike+codelen (inf where no fit).
    lib  : (N_funcs,) one-time structural/library cost per function.
    init : (N_clusters,) initial assignment (func index per cluster); default greedy.
    Reassigns each cluster to the function minimising the *total* DL given the
    current counts. Iterates to a fixed point. Returns the assignment + costs.
    """
    L = np.asarray(L, dtype=float)
    lib = np.asarray(lib, dtype=float)
    N, F = L.shape

    if init is None:
        assign = np.nanargmin(np.where(np.isfinite(L), L, np.inf), axis=1)
    else:
        assign = np.array(init, dtype=int)

    def total_dl(a):
        counts = np.bincount(a, minlength=F)
        used = counts > 0
        fit = float(L[np.arange(N), a].sum())
        libc = float(lib[used].sum())
        ac = assignment_cost(counts[used].tolist(), assignment)
        return fit + libc + ac

    cur = total_dl(assign)
    for _ in range(max_iter):
        moved = False
        order = np.arange(N)
        if rng is not None:
            rng.shuffle(order)
        for i in order:
            best_f = assign[i]
            best_cost = cur
            for f in range(F):
                if f == assign[i] or not np.isfinite(L[i, f]):
                    continue
                trial = assign.copy()
                trial[i] = f
                c = total_dl(trial)
                if c < best_cost - 1e-12:
                    best_cost, best_f = c, f
            if best_f != assign[i]:
                assign[i] = best_f
                cur = best_cost
                moved = True
        if not moved:
            break

    counts = np.bincount(assign, minlength=F)
    used = counts > 0
    return {
        "assign": assign,
        "M": int(used.sum()),
        "fit": float(L[np.arange(N), assign].sum()),
        "lib": float(lib[used].sum()),
        "assign_cost": assignment_cost(counts[used].tolist(), assignment),
        "total": cur,
        "counts": counts,
    }


def coordinate_descent_multistart(L, lib, assignment="entropy",
                                  n_restarts=8, seed=0):
    """Run coordinate_descent from several inits; keep the best.

    Starting points:
      1. greedy (each cluster on its own single-best function),
      2. every "all clusters on function f" for f that fits ALL clusters -- these
         are exactly the universal (M=1) solutions, so per-cluster can NEVER do
         worse than the best universal profile (guards the local minimum where a
         fragmented init can't collapse back to a single function),
      3. n_restarts random inits.
    """
    L = np.asarray(L, dtype=float)
    N, F = L.shape
    rng = np.random.default_rng(seed)

    log = []          # one entry per restart: kind, total, M, counts, is_best
    best = None

    def run(kind, init):
        nonlocal best
        res = coordinate_descent(L, lib, assignment, init=init, rng=rng)
        is_best = best is None or res["total"] < best["total"] - 1e-12
        if is_best:
            best = res
        log.append({"kind": kind, "total": res["total"], "M": res["M"],
                    "counts": res["counts"].copy(), "is_best": is_best})

    run("greedy", None)

    # Universal seeds: everyone on one function (only functions finite for all).
    for f in np.where(np.isfinite(L).all(axis=0))[0]:
        run(f"universal[f={f}]", np.full(N, f, dtype=int))

    for r in range(n_restarts):
        init = rng.integers(0, F, size=N)
        # repair inits that land on an invalid (inf) fit
        for i in range(N):
            if not np.isfinite(L[i, init[i]]):
                finite = np.where(np.isfinite(L[i]))[0]
                init[i] = finite[0] if finite.size else 0
        run(f"random[{r}]", init)

    best["restart_log"] = log
    return best


# ==========================================
# I/O  (CLASH: match in all_equations; best-funcs source)
# ==========================================

def load_top_functions(path, n=TOP_N):
    raw = np.genfromtxt(path, dtype=str, delimiter="\t", skip_header=1, comments=None)
    out = []
    for i in range(min(n, len(raw))):
        out.append({"func": raw[i, 1],
                    "ay": float(raw[i, 6]) if raw[i, 6] != "nan" else 0.0,
                    "comp": int(raw[i, -1])})
    return out


# ---- cached per-(source, comp) lookup table ----
_unique_cache = {}   # (source, comp) -> {canonical_key: [all-equations row indices]}


def make_canon():
    """SymPy canonical key, identical to build_full_clash_ranking.py.

    Params are Abs-wrapped -> positive symbols, then powsimp/powdenest. This
    collapses reparametrised / simplified variants (e.g. pow(Abs(a1),x),
    x*(-a0+x)-x) onto the same key as their exhaustive-library form. Falls back
    to a whitespace-stripped raw string if SymPy is unavailable or the
    expression won't parse (so nothing crashes; those just match literally).
    """
    try:
        import sympy as sp
    except Exception:  # noqa: BLE001 - SymPy optional; degrade to literal match
        print("[warn] SymPy not available - matching falls back to literal string "
              "compare; reparametrised/simplified functions will NOT resolve.")
        canon = lambda f: f.replace(" ", "").strip()  # noqa: E731
        canon.has_sympy = False
        return canon

    x = sp.Symbol("x", positive=True)
    loc = {"x": x, "Abs": sp.Abs}
    for i in range(8):
        loc[f"a{i}"] = sp.Symbol(f"a{i}", positive=True)
    cache = {}

    def canon(f):
        if f in cache:
            return cache[f]
        try:
            e = sp.sympify(f, locals=loc)
            e = sp.powdenest(sp.powsimp(e, force=True), force=True)
            key = sp.srepr(e)
        except Exception as exc:  # noqa: BLE001 - unparseable -> unique to itself
            key = "RAW::" + f.replace(" ", "").strip()
            canon._parse_fail[f] = str(exc)
        cache[f] = key
        return key
    canon.has_sympy = True
    canon._parse_fail = {}   # func string -> sympify error, for diagnostics
    return canon


_canon = make_canon()


def _alleq_map(source, comp):
    """canonical key -> list of ALL-EQUATIONS row indices for that class.

    Row i in all_equations_{comp}.txt is the specific variant ESR generated, and
    it lines up 1:1 with row i of codelen_matches_comp{comp}.dat. We do NOT go
    through unique_equations/matches: unique_equations stores ONE canonical
    variant per class, which need not be the variant a given ranking-table string
    (or the actual fitted row) uses. Canonicalising every all-eq row and grouping
    by key lets any spelling of a function find all of its fitted rows.
    """
    key = (source, comp)
    if key not in _unique_cache:
        comp_dir = Path(SOURCES[source]["lib"]) / f"compl_{comp}"
        cands = sorted(comp_dir.glob("all_equations*.txt"))
        d = {}
        if cands:
            with open(cands[0], encoding="utf-8") as f:
                for i, line in enumerate(f):
                    # all_equations rows may be "idx;expr" or bare "expr"
                    expr = line.strip().split(";")[-1].strip()
                    if not expr:
                        continue
                    d.setdefault(_canon(expr), []).append(i)
        _unique_cache[key] = d
    return _unique_cache[key]


def resolve_function(fd):
    """Attach the fit source + all-equations rows for a function.

    Tries SOURCE_ORDER: canonicalise the function and look it up among the
    canonicalised all_equations rows. Stores fd['source'] and fd['rows'] (every
    all-eq row i whose canonical form equals this function's). Returns True if
    resolved. On failure sets fd['fail_reason']: 'parse' (SymPy couldn't sympify
    it), 'missing_lib' (all_equations file absent for that complexity),
    'not_in_library' (canonical form not found among the all-eq rows).
    """
    target = _canon(fd["func"])
    if getattr(_canon, "has_sympy", True) and target.startswith("RAW::"):
        fd["fail_reason"] = "parse"
        return False
    reason = "not_in_library"
    for source in SOURCE_ORDER:
        amap = _alleq_map(source, fd["comp"])
        if not amap:
            reason = "missing_lib"
            continue
        rows = amap.get(target)
        if rows:
            fd["source"] = source
            fd["rows"] = np.array(rows, dtype=int)
            return True
    fd["fail_reason"] = reason
    return False


def get_cluster_metrics(cluster_name, fd):
    """(negloglike, codelen) for function fd on one cluster, from fd['source'].

    Reads codelen_matches_comp{comp}.dat (col0 negloglike, col1 codelen; row i
    lines up with all_equations row i). Takes the best (min negll+codelen) over
    fd['rows'] (all all-eq rows whose canonical form is this function). Returns
    None if unavailable.
    """
    suffix = SOURCES[fd["source"]]["suffix"]
    fp = (Path(FIT_ROOT_DIR) / f"output_WL_{cluster_name}{suffix}"
          / f"codelen_matches_comp{fd['comp']}.dat")
    if not fp.exists():
        return None
    try:
        data = np.genfromtxt(fp)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        rows = fd["rows"][fd["rows"] < len(data)]
        if rows.size == 0:
            return None
        sub = data[rows]
        totals = sub[:, 0] + sub[:, 1]
        finite = np.isfinite(totals)
        if not finite.any():
            return None
        best = sub[np.where(finite)[0][np.argmin(totals[finite])]]
        return float(best[0]), float(best[1])   # (negloglike, codelen)
    except Exception as e:  # noqa: BLE001 - keep scanning other clusters
        print(e)
        return None


def build_L_matrix(funcs, clusters):
    """Return (L, NEG, COD): three (N_clusters, N_funcs) matrices.

    L = negloglike + codelen (the object minimised); NEG, COD are the components
    (inf where no fit). Each function's fit comes from its resolved source.
    """
    N, F = len(clusters), len(funcs)
    NEG = np.full((N, F), np.inf)
    COD = np.full((N, F), np.inf)
    for j, fd in enumerate(funcs):
        for i, clust in enumerate(clusters):
            m = get_cluster_metrics(clust, fd)
            if m is not None:
                NEG[i, j], COD[i, j] = m[0], m[1]
    return NEG + COD, NEG, COD


# ==========================================
# SELF-TEST  (runs locally, no data files)
# ==========================================

def _brute_force_pair(La, Lb, lib_a, lib_b, assignment):
    """Exhaustive 2^N reference optimum for validating the sweep (small N only)."""
    N = len(La)
    best = math.inf
    for mask in range(1 << N):
        n_a = bin(mask).count("1")
        n_b = N - n_a
        fit = 0.0
        ok = True
        for i in range(N):
            side_a = (mask >> i) & 1
            Li = La[i] if side_a else Lb[i]
            if math.isinf(Li):
                ok = False
                break
            fit += Li
        if not ok:
            continue
        lib = (lib_a if n_a > 0 else 0.0) + (lib_b if n_b > 0 else 0.0)
        assign = assignment_cost([n_a, n_b], assignment)
        best = min(best, fit + lib + assign)
    return best


def self_test():
    rng = np.random.default_rng(42)
    print("== self-test: assignment cost matches paper Eq.(34) ==")
    # paper: rho2 chosen by 62 clusters, rho3 by 87 -> L_assign = 101.17 nats
    got = assignment_cost([62, 87], "entropy")
    print(f"   62/87 split -> {got:.2f} nats (paper: 101.17)")
    assert abs(got - 101.17) < 0.02, got
    assert assignment_cost([149], "entropy") == 0.0  # single profile -> no cost

    print("== self-test: sweep == brute force, and sweep <= greedy ==")
    # (require_both=False here so the sweep spans the SAME space as brute force,
    #  which enumerates all 2^N splits including degenerate ones.)
    for trial in range(200):
        N = rng.integers(3, 13)
        La = rng.normal(50, 10, size=N)
        Lb = rng.normal(50, 10, size=N)
        lib_a, lib_b = rng.uniform(5, 30, size=2)
        # occasionally kill a fit on one side
        if rng.random() < 0.3:
            La[rng.integers(N)] = np.inf
        opt = optimal_pair(La, Lb, lib_a, lib_b, require_both=False)["total"]
        grd = greedy_pair(La, Lb, lib_a, lib_b)["total"]
        bf = _brute_force_pair(La.tolist(), Lb.tolist(), lib_a, lib_b, "entropy")
        assert abs(opt - bf) < 1e-6, (trial, opt, bf)      # exact
        assert opt <= grd + 1e-9, (trial, opt, grd)        # never worse than greedy
    print("   200 random pairs: sweep == brute force, sweep <= greedy  OK")

    print("== self-test: require_both forbids n=0 splits ==")
    # 4 clusters, all cleanly prefer B. With require_both=False the optimum puts
    # all 4 on B (n_a=0). With require_both=True that split is illegal, so n_a>=1.
    La = np.array([20., 21., 22., 23.]); Lb = np.array([9., 9., 9., 9.])
    free = optimal_pair(La, Lb, 15., 15., require_both=False)
    forced = optimal_pair(La, Lb, 15., 15., require_both=True)
    print(f"   free: n_a={free['n_a']} n_b={free['n_b']}  |  "
          f"require_both: n_a={forced['n_a']} n_b={forced['n_b']}")
    assert free["n_a"] == 0                       # unconstrained collapses to B
    assert forced["n_a"] >= 1 and forced["n_b"] >= 1   # forced keeps both
    assert forced["total"] >= free["total"] - 1e-9     # constraint can't help
    # if a two-function split is impossible, returns None:
    La2 = np.array([10., np.inf]); Lb2 = np.array([np.inf, 10.])  # each cluster one-sided
    assert optimal_pair(La2, Lb2, 5., 5., require_both=True) is not None  # 1/1 split ok
    La3 = np.array([10., 11.]); Lb3 = np.array([np.inf, np.inf])   # only A can fit either
    assert optimal_pair(La3, Lb3, 5., 5., require_both=True) is None

    print("== self-test: a case where global STRICTLY beats greedy ==")
    # 4 clusters. Clusters 0-2 clearly prefer B; cluster 3 barely prefers A
    # (9.8 < 10.0). Greedy sends cluster 3 to A, paying lib(A) + splitting the
    # sample (nonzero L_assign). Keeping all four on B (drop lib(A), L_assign=0)
    # is globally cheaper despite cluster 3's 0.2-nat worse fit.
    # (require_both=False: this demonstration is specifically about the n=0 win.)
    La = np.array([10.0, 10.0, 10.0, 9.8])
    Lb = np.array([9.0, 9.0, 9.0, 10.0])
    lib_a, lib_b = 15.0, 15.0
    g = greedy_pair(La, Lb, lib_a, lib_b)
    o = optimal_pair(La, Lb, lib_a, lib_b, require_both=False)
    print(f"   greedy total={g['total']:.3f} (split {g['n_a']}/{g['n_b']}), "
          f"global total={o['total']:.3f} (split {o['n_a']}/{o['n_b']})")
    assert o["total"] <= g["total"] + 1e-9

    print("== self-test: coordinate descent <= its greedy init, on a 3-func pool ==")
    N, F = 30, 3
    L = rng.normal(50, 8, size=(N, F))
    lib = rng.uniform(5, 25, size=F)
    greedy_assign = np.argmin(L, axis=1)
    gc = np.bincount(greedy_assign, minlength=F)
    greedy_total = (L[np.arange(N), greedy_assign].sum()
                    + lib[gc > 0].sum()
                    + assignment_cost(gc[gc > 0].tolist(), "entropy"))
    cd = coordinate_descent_multistart(L, lib, n_restarts=6, seed=1)
    print(f"   greedy-init total={greedy_total:.3f}, CD best={cd['total']:.3f}, "
          f"M={cd['M']}")
    assert cd["total"] <= greedy_total + 1e-9

    print("== self-test: three-model ordering on a self-similar synthetic sample ==")
    # Build a sample where one function fits everyone well (a shared/universal
    # profile) so M=1 SHOULD win once assignment+structure costs are counted -
    # a synthetic stand-in for the paper's self-similar-haloes result.
    N, F = 40, 5
    L = rng.uniform(60, 90, size=(N, F))
    L[:, 0] = rng.uniform(20, 24, size=N)   # func 0: uniformly great for all clusters
    lib = rng.uniform(5, 25, size=F)
    m1 = universal_model(L, lib)
    m2, _ = best_pair_model(L, lib)                       # require_both=True by default
    mN = per_cluster_model(L, lib, n_restarts=6)
    print(f"   M=1 total={m1['total']:.2f}  M=2 total={m2['total']:.2f} "
          f"(n={m2['n_a']}/{m2['n_b']})  per-cluster total={mN['total']:.2f} (M={mN['M']})")
    assert m1["assign_cost"] == 0.0                       # universal has no label cost
    assert m2["n_a"] >= 1 and m2["n_b"] >= 1              # M=2 is a genuine two-func split
    assert mN["total"] <= mN["greedy_total"] + 1e-9       # global per-cluster <= greedy
    assert m1["total"] <= m2["total"] + 1e-9              # M=2 forced-split can't beat M=1 here
    # KEY INVARIANT: per-cluster has strictly MORE freedom than universal (the
    # all-on-one-function assignments are in its search space), so it can never
    # score worse. mN <= m1, NOT the other way round.
    assert mN["total"] <= m1["total"] + 1e-9

    print("== self-test: per-cluster never worse than universal (local-min guard) ==")
    # Reproduces the reported bug: several functions each fit all clusters well;
    # a fragmented random init could strand coordinate descent ABOVE the best
    # single-function solution. The universal-seed starts must prevent that.
    rng2 = np.random.default_rng(7)
    worst_margin = 0.0
    for _ in range(30):
        N, F = 20, 6
        L = rng2.uniform(40, 90, size=(N, F))
        # make 3 functions each a near-universal good fit (so M=1 is attractive)
        for f in range(3):
            L[:, f] = rng2.uniform(6.0 + f, 6.5 + f, size=N)
        lib = rng2.uniform(5, 25, size=F)
        u = universal_model(L, lib)
        pc = per_cluster_model(L, lib, n_restarts=6)
        # per-cluster must be <= universal, always
        assert pc["total"] <= u["total"] + 1e-9, (pc["total"], u["total"])
        worst_margin = max(worst_margin, u["total"] - pc["total"])
    print(f"   30 cases: per-cluster <= universal always "
          f"(best improvement seen: {worst_margin:.2f} nats)")

    print("== self-test: greedy pair model (each cluster picks its better) ==")
    # 6 clusters, 3 funcs. For a fixed pair, greedy assigns each cluster to its
    # smaller L; check the best greedy pair's split matches a hand computation.
    L = np.array([[10, 20, 50],
                  [10, 20, 50],
                  [30,  8, 50],
                  [30,  8, 50],
                  [12, 40, 50],
                  [40, 15, 50]], float)
    lib = np.array([5., 6., 7.])
    m2g, grows = best_greedy_pair_model(L, lib)
    # for pair (0,1): clusters 0,1,4 -> A (col0 smaller); 2,3,5 -> B -> 3/3 split
    p01 = next(r for r in grows if (r["a"], r["b"]) == (0, 1))
    assert p01["n_a"] == 3 and p01["n_b"] == 3, (p01["n_a"], p01["n_b"])
    exp_fit = L[[0, 1, 4], 0].sum() + L[[2, 3, 5], 1].sum()
    assert abs(p01["fit"] - exp_fit) < 1e-9, (p01["fit"], exp_fit)
    assert m2g["n_a"] >= 1 and m2g["n_b"] >= 1        # genuine two-function split
    print(f"   best greedy pair total={m2g['total']:.2f}, split "
          f"{m2g['n_a']}/{m2g['n_b']}  OK")

    print("== self-test: full-library per-cluster merges base run + best re-run ==")
    import tempfile
    tmp = Path(tempfile.mkdtemp())
    # cX: base has funcA (negll+cod=15) and funcB (25); the re-run has funcA with
    #     a BETTER fit (8+2=10) -> full-lib must use the re-run's funcA (fit 10).
    # cY: base only (funcC best at 5+2=7); no re-run folder -> falls back to base.
    (tmp / "output_WL_cX").mkdir(parents=True)
    (tmp / "output_WL_cX" / "final_5.dat").write_text(
        "0;funcA;30;x;10;5;3\n0;funcB;40;x;20;5;4\n")     # base: A=15, B=25
    (tmp / "output_WL_cX_RR").mkdir(parents=True)
    (tmp / "output_WL_cX_RR" / "final_5.dat").write_text(
        "0;funcA;12;x;8;2;3\n")                           # re-run: A=10 (better)
    (tmp / "output_WL_cY").mkdir(parents=True)
    (tmp / "output_WL_cY" / "final_5.dat").write_text(
        "0;funcA;30;x;12;6;3\n0;funcC;20;x;5;2;4\n")      # base: C=7 wins
    old_root, old_suf = FIT_ROOT_DIR, BEST_RERUN_SUFFIX
    globals()["FIT_ROOT_DIR"] = str(tmp)
    globals()["BEST_RERUN_SUFFIX"] = "_RR"
    try:
        mf = full_library_per_cluster_model(["cX", "cY"])
    finally:
        globals()["FIT_ROOT_DIR"] = old_root
        globals()["BEST_RERUN_SUFFIX"] = old_suf
    fX, nX, cX_, sX = mf["chosen"]["cX"]
    fY, nY, cY_, sY = mf["chosen"]["cY"]
    assert fX == "funcA" and sX == "rerun", mf["chosen"]["cX"]   # re-run fit used
    assert (nX, cX_) == (8.0, 2.0), (nX, cX_)                    # the improved values
    assert fY == "funcC" and sY == "base", mf["chosen"]["cY"]    # base fallback
    # fit = cX rerun (8+2) + cY base (5+2) = 17
    assert abs(mf["fit"] - 17.0) < 1e-9, mf["fit"]
    assert mf["n_from_rerun"] == 1 and mf["M"] == 2 and mf["N"] == 2
    print(f"   full-lib: cX->funcA(rerun) cY->funcC(base), fit={mf['fit']:.1f}, "
          f"{mf['n_from_rerun']} from re-run  OK")

    print("\nALL SELF-TESTS PASSED")


# ==========================================
# THE MODELS  (paper section 5.4 comparison, one consistent cost)
# ==========================================

def universal_model(L, lib, assignment="entropy"):
    """M=1: every cluster on one shared function. L_assign = 0 by construction.

    Pick the single column minimising total fit + that function's one library
    cost. A function is eligible ONLY if EVERY cluster has a finite fit for it -
    a universal profile must describe the whole sample. Returns None if no
    function fits all clusters (the caller reports this and stops).
    """
    N, F = L.shape
    eligible = np.isfinite(L).all(axis=0)
    best = None
    for f in np.where(eligible)[0]:
        fit = float(L[:, f].sum())
        total = fit + float(lib[f])           # assignment cost is 0 for M=1
        if best is None or total < best["total"]:
            best = {"func": f, "M": 1, "fit": fit, "lib": float(lib[f]),
                    "assign_cost": 0.0, "total": total,
                    "counts": np.eye(F, dtype=int)[f] * N}
    return best


def _pair_assignment(La, Lb, cut_k):
    """Reconstruct the per-cluster A/B assignment for a pair at its optimal cut.

    Mirrors optimal_pair: forced clusters (one side inf) go to their only side;
    the free clusters (finite on both) are sorted by gap and the cut_k smallest
    go to A. Returns a boolean mask `to_a` over the ORIGINAL cluster order, plus
    the kept mask (drops clusters that are inf on both sides).
    """
    La = np.asarray(La, float); Lb = np.asarray(Lb, float)
    only_b = np.isinf(La) & np.isfinite(Lb)
    only_a = np.isinf(Lb) & np.isfinite(La)
    both = np.isfinite(La) & np.isfinite(Lb)
    keep = ~(np.isinf(La) & np.isinf(Lb))
    to_a = np.zeros(len(La), dtype=bool)
    to_a[only_a] = True
    idx = np.where(both)[0]
    order = idx[np.argsort((La - Lb)[idx], kind="mergesort")]
    to_a[order[:cut_k]] = True     # cut_k smallest gaps -> A; rest stay False (B)
    return to_a, keep


def best_pair_model(L, lib, assignment="entropy", NEG=None, COD=None):
    """M=2: exact sweep over every pair; return the best pair and the full table.

    If NEG/COD are given, each row also carries the negloglike/codelen breakdown
    of its optimal split (for the detailed all-pairs table).
    """
    F = L.shape[1]
    rows = []
    for a, b in itertools.combinations(range(F), 2):
        o = optimal_pair(L[:, a], L[:, b], lib[a], lib[b], assignment)
        if o is None:               # no genuine two-function split for this pair
            continue
        g = greedy_pair(L[:, a], L[:, b], lib[a], lib[b], assignment)
        row = {"a": a, "b": b, **o, "total_greedy": g["total"]}
        # Which clusters sit on each side of the optimal split, and by how much
        # each cluster prefers its side (gap = L_a - L_b; negative -> prefers A).
        to_a, keep = _pair_assignment(L[:, a], L[:, b], o["cut_k"])
        row["to_a"], row["keep"] = to_a, keep
        row["gap"] = L[:, a] - L[:, b]          # per-cluster fit-gap
        if NEG is not None:
            neg = float(NEG[to_a & keep, a].sum() + NEG[(~to_a) & keep, b].sum())
            cod = float(COD[to_a & keep, a].sum() + COD[(~to_a) & keep, b].sum())
            row["neg"], row["cod"] = neg, cod
        rows.append(row)
    if not rows:
        return None, []             # no valid two-function split anywhere
    rows.sort(key=lambda r: r["total"])
    best = rows[0]
    return {"M": 2, "func_a": best["a"], "func_b": best["b"],
            "fit": best["fit"], "lib": best["lib"],
            "assign_cost": best["assign"], "total": best["total"],
            "total_greedy": best["total_greedy"],
            "n_a": best["n_a"], "n_b": best["n_b"],
            "to_a": best["to_a"], "keep": best["keep"], "gap": best["gap"]}, rows


def best_greedy_pair_model(L, lib, assignment="entropy", NEG=None, COD=None):
    """M=2 GREEDY: the old all_two_models_2.py method. For each pair, every cluster
    picks whichever of the two functions it fits better (argmin L per cluster);
    the library + assignment costs are added AFTER. Returns the best greedy pair
    and the full per-pair table. This is the local-choice baseline the global
    sweep improves on - reported as its own model-comparison row."""
    F = L.shape[1]
    rows = []
    for a, b in itertools.combinations(range(F), 2):
        g = greedy_pair(L[:, a], L[:, b], lib[a], lib[b], assignment)
        # skip degenerate greedy splits so this row is a genuine two-function model
        if g["n_a"] == 0 or g["n_b"] == 0:
            continue
        to_a = L[:, a] <= L[:, b]
        keep = np.isfinite(L[:, a]) | np.isfinite(L[:, b])
        row = {"a": a, "b": b, **g, "to_a": to_a, "keep": keep,
               "gap": L[:, a] - L[:, b]}
        if NEG is not None:
            row["neg"] = float(NEG[to_a & keep, a].sum() + NEG[(~to_a) & keep, b].sum())
            row["cod"] = float(COD[to_a & keep, a].sum() + COD[(~to_a) & keep, b].sum())
        rows.append(row)
    if not rows:
        return None, []
    rows.sort(key=lambda r: r["total"])
    best = rows[0]
    return {"M": 2, "func_a": best["a"], "func_b": best["b"],
            "fit": best["fit"], "lib": best["lib"],
            "assign_cost": best["assign"], "total": best["total"],
            "n_a": best["n_a"], "n_b": best["n_b"],
            "to_a": best["to_a"], "keep": best["keep"], "gap": best["gap"]}, rows


def per_cluster_model(L, lib, assignment="entropy", n_restarts=8):
    """Full per-cluster model (paper section 5.4.1): each cluster free to pick ANY
    function. This is coordinate descent over the whole pool minimising the TOTAL
    DL - the globally-optimised counterpart of different_models_2.py's greedy
    per-cluster choice. Also reports the pure greedy total for comparison."""
    N, F = L.shape
    greedy_assign = np.nanargmin(np.where(np.isfinite(L), L, np.inf), axis=1)
    gc = np.bincount(greedy_assign, minlength=F)
    greedy_total = (float(L[np.arange(N), greedy_assign].sum())
                    + float(lib[gc > 0].sum())
                    + assignment_cost(gc[gc > 0].tolist(), assignment))
    cd = coordinate_descent_multistart(L, lib, assignment, n_restarts=n_restarts)
    cd["greedy_total"] = greedy_total
    cd["greedy_M"] = int((gc > 0).sum())
    return cd


def _read_final_dats(folder):
    """Read a cluster's final_*.dat files in one folder -> {func: (dl_no_aif,
    negll, cod, aif)} keeping the best (lowest negll+cod) per function.

    ALL equations are listed in these files (not just line 1), so we scan every
    line, not only the first. Line format (';'-delimited): parts[1]=func,
    parts[4]=negloglike, parts[5]=codelen, parts[6]=aifeyn.
    """
    out = {}
    if not folder.is_dir():
        return out
    for fp in folder.glob("final_*.dat"):
        try:
            with open(fp) as f:
                lines = f.readlines()
        except OSError:
            continue
        for line in lines:
            parts = line.strip().split(";")
            if len(parts) < 7:
                continue
            try:
                func = parts[1]
                negll = float(parts[4]); cod = float(parts[5]); aif = float(parts[6])
            except ValueError:
                continue
            dl = negll + cod
            cur = out.get(func)
            if cur is None or dl < cur[0]:
                out[func] = (dl, negll, cod, aif)
    return out


def full_library_per_cluster_model(clusters, assignment="entropy"):
    """Per-cluster model drawing from the WHOLE library, not the top-N pool.

    For each cluster, pool its fits from BOTH runs:
      * the base run (output_WL_<cluster>), which lists ALL equations, and
      * the best-funcs re-run (BEST_RERUN_SUFFIX), a subset run with more
        iterations / snapping -> better convergence.
    For each function we take the better (lower negll+cod) of the two runs, so a
    function present in the re-run uses the re-run's improved fit; functions only
    in the base run fall back to it. Each cluster then picks its single overall
    best function. Library cost is paid once per distinct function used, plus the
    assignment cost.
    """
    total_neg = total_cod = 0.0
    chosen = {}                       # cluster -> (func, negll, cod, source)
    func_cost = {}                    # func -> aifeyn (paid once)
    counts = {}
    missing = []
    for name in clusters:
        base = _read_final_dats(Path(FIT_ROOT_DIR) / f"output_WL_{name}")
        rerun = _read_final_dats(Path(FIT_ROOT_DIR) / f"output_WL_{name}{BEST_RERUN_SUFFIX}")
        if not base and not rerun:
            missing.append(name)
            continue
        # merge: re-run wins for a function unless the base fit is strictly better
        merged = dict(base)                     # func -> (dl, negll, cod, aif)
        src = {f: "base" for f in base}
        for func, v in rerun.items():
            if func not in merged or v[0] < merged[func][0]:
                merged[func] = v
                src[func] = "rerun"
        # this cluster's single best function across the whole merged library
        func = min(merged, key=lambda f: merged[f][0])
        dl, negll, cod, aif = merged[func]
        total_neg += negll; total_cod += cod
        chosen[name] = (func, negll, cod, src[func])
        func_cost.setdefault(func, aif)
        counts[func] = counts.get(func, 0) + 1

    N = len(chosen)
    if N == 0:
        return None
    library_cost = float(sum(func_cost.values()))
    assign_cost = assignment_cost(list(counts.values()), assignment)
    fit = total_neg + total_cod
    n_rerun = sum(1 for v in chosen.values() if v[3] == "rerun")
    return {
        "M": len(func_cost), "fit": fit, "lib": library_cost,
        "assign_cost": assign_cost, "total": fit + library_cost + assign_cost,
        "chosen": chosen, "counts": counts, "missing": missing, "N": N,
        "n_from_rerun": n_rerun,
    }


# ==========================================
# MAIN
# ==========================================

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true",
                    help="run synthetic unit tests (no data files needed)")
    ap.add_argument("--assignment", choices=["entropy", "enumerative"],
                    default="entropy", help="assignment cost model (default: entropy, = paper Eq.34)")
    ap.add_argument("--top-n", type=int, default=TOP_N)
    ap.add_argument("--restarts", type=int, default=8,
                    help="random restarts for the per-cluster coordinate descent")
    ap.add_argument("--restart-log", action="store_true",
                    help="print where every coordinate-descent restart landed "
                         "(kind, total DL, M) - shows the local-minimum landscape")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    # --- load + resolve + build L matrix (the one expensive I/O pass) ---
    funcs = load_top_functions(COMBINE_FILE, args.top_n)
    with open(CLUSTERS_LIST) as f:
        clusters = [ln.strip() for ln in f if ln.strip()]

    reason_txt = {
        "parse": "SymPy could not parse it (RAW:: key)",
        "missing_lib": "unique_equations_{comp}.txt not found for its complexity",
        "not_in_library": "canonical form not among the all_equations rows for its complexity",
    }
    print(f"SymPy canonicalisation: {'ON' if getattr(_canon, 'has_sympy', False) else 'OFF (literal match)'}")
    valid = []
    for i, fd in enumerate(funcs):
        if resolve_function(fd):        # sets fd['source'], fd['u'], fd['rows']
            fd["original_id"] = i
            valid.append(fd)
        else:
            why = reason_txt.get(fd.get("fail_reason", ""), fd.get("fail_reason", "?"))
            print(f"Skipping function {i} [comp {fd['comp']}] - {why}: {fd['func'][:50]}")

    if getattr(_canon, "has_sympy", False) and _canon._parse_fail:
        print(f"[warn] SymPy failed to parse {len(_canon._parse_fail)} string(s); "
              f"they can only match literally. First few:")
        for k in list(_canon._parse_fail)[:5]:
            print(f"         {k}  ({_canon._parse_fail[k]})")

    print(f"\nAssignment cost: {args.assignment}  |  source: best-funcs run")
    print(f"Resolved {len(valid)}/{len(funcs)} functions in the best-funcs library.")
    print(f"Building L-matrix: {len(clusters)} clusters x {len(valid)} functions ...")
    L, NEG, COD = build_L_matrix(valid, clusters)

    # Sanity: per-cluster fit magnitude (compare to the paper's ~5.5 nats/cluster).
    fin = np.isfinite(L)
    if fin.any():
        print(f"[info] finite L cells: mean={L[fin].mean():.2f}, "
              f"min={L[fin].min():.2f}, max={L[fin].max():.2f} nats "
              f"(per (cluster,function)).")
    lib = np.array([fd["ay"] for fd in valid])
    ids = [fd["original_id"] for fd in valid]
    func_str = {j: valid[j]["func"] for j in range(len(valid))}

    # --- L-matrix diagnostics: where are the missing fits? ---
    finite = np.isfinite(L)
    n_missing = int((~finite).sum())
    if n_missing:
        print(f"\n[warn] L-matrix has {n_missing}/{L.size} missing (inf) cells.")
        per_clust = (~finite).sum(axis=1)
        dead_clust = np.where(per_clust == L.shape[1])[0]
        if dead_clust.size:
            print(f"       {dead_clust.size} cluster(s) have NO valid fit for ANY "
                  f"function (dropped from every model):")
            for i in dead_clust:
                print(f"         - {clusters[i]}")
        per_func = (~finite).sum(axis=0)
        for j in np.where(per_func > 0)[0]:
            print(f"       func [{ids[j]:>2}] missing for {per_func[j]}/{L.shape[0]} "
                  f"clusters: {func_str[j][:50]}")

    # Drop clusters with no valid fit for any function - they carry no information
    # and would make every model ill-defined (this is what crashed universal_model).
    keep = finite.any(axis=1)
    if not keep.all():
        L, NEG, COD = L[keep], NEG[keep], COD[keep]
        clusters = [c for c, k in zip(clusters, keep) if k]
        print(f"[info] proceeding with {L.shape[0]} clusters that have >=1 valid fit.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- the three models, one cost model ---
    m1 = universal_model(L, lib, args.assignment)
    if m1 is None:
        n_full = int(np.isfinite(L).all(axis=0).sum())
        print(f"[error] no single function has a finite fit for all {L.shape[0]} "
              f"clusters ({n_full} functions qualify), so there is no universal "
              f"M=1 profile. Check the L-matrix diagnostics above - the missing "
              f"cells are the blocker.")
        return
    m2, pair_rows = best_pair_model(L, lib, args.assignment, NEG, COD)
    m2g, greedy_pair_rows = best_greedy_pair_model(L, lib, args.assignment, NEG, COD)
    mN = per_cluster_model(L, lib, args.assignment, args.restarts)
    mFull = full_library_per_cluster_model(clusters, args.assignment)

    # Save the full all-pairs tables (best first) with every term broken out, the
    # way the old all_two_models_2.py did: NegL, CodeL, their sum, library cost,
    # assignment cost, total, split counts, and the clusters on the LEAST-USED
    # function of the split (keyed by that function's ID).
    def least_used_clusters(r):
        """'<least_func_ID>: name1,name2,...' for the pair's less-populated func."""
        on_a = np.where(r["to_a"] & r["keep"])[0]
        on_b = np.where((~r["to_a"]) & r["keep"])[0]
        if len(on_b) <= len(on_a):
            least_rows, least_id = on_b, ids[r["b"]]
        else:
            least_rows, least_id = on_a, ids[r["a"]]
        names = ",".join(clusters[i] for i in least_rows)
        return f"{least_id}: {names}"

    def write_pairs_table(path, rows, greedy=False):
        with open(path, "w") as f:
            hdr = ("{:^5}{:^5}|{:^42}{:^42}|{:^12}{:^12}{:^12}{:^12}{:^12}"
                   "{:^13}{:^7}{:^7}  {}").format(
                "ID_A", "ID_B", "func_A", "func_B", "NegL", "CodeL",
                "NegL+CodeL", "Lib", "Assign", "Total_DL",
                "n_A", "n_B", "least_used_func_ID: clusters")
            f.write(hdr + "\n" + "-" * len(hdr) + "\n")
            for r in rows:
                f.write("{:^5}{:^5}|{:^42}{:^42}|{:^12.3f}{:^12.3f}{:^12.3f}"
                        "{:^12.3f}{:^12.3f}{:^13.3f}{:^7}{:^7}  {}\n".format(
                    ids[r['a']], ids[r['b']],
                    func_str[r['a']][:40], func_str[r['b']][:40],
                    r['neg'], r['cod'], r['fit'], r['lib'], r['assign'],
                    r['total'], r['n_a'], r['n_b'], least_used_clusters(r)))

    out_pairs = OUTPUT_DIR / "two_func_pairs_global.txt"
    write_pairs_table(out_pairs, pair_rows)
    print(f"[info] wrote all {len(pair_rows)} GLOBAL pairs -> {out_pairs}")

    out_pairs_greedy = OUTPUT_DIR / "two_func_pairs_greedy.txt"
    if greedy_pair_rows:
        write_pairs_table(out_pairs_greedy, greedy_pair_rows)
        print(f"[info] wrote all {len(greedy_pair_rows)} GREEDY pairs -> {out_pairs_greedy}")

    # --- model-comparison summary (one consistent cost model) ---
    def fit_of(model):   # residuals+parameters term (sum over clusters)
        return model["fit"]

    # Rows we rank together. ALL of these read fit from the SAME source
    # (codelen_matches / the L-matrix), so their Fit columns ARE comparable.
    # full-lib is NOT here - it reads final_*.dat, a different file - so it is
    # reported separately below to avoid a bogus cross-source Fit comparison.
    model_rows = [("Universal (M=1)", 1, m1)]
    if m2 is not None:
        model_rows.append(("Two profiles (global)", 2, m2))
    if m2g is not None:
        model_rows.append(("Two profiles (greedy)", 2, m2g))
    model_rows.append((f"Per-cluster (top-{len(valid)})", mN["M"], mN))
    winner_total = min(m["total"] for _, _, m in model_rows)

    table_path = OUTPUT_DIR / "model_comparison.txt"
    with open(table_path, "w") as f:
        hdr = (f"{'':1}{'Model':<23}{'M':>4}{'Fit (resid+param)':>20}"
               f"{'Function cost':>16}{'Assignment':>14}{'Total L(D)':>14}\n")
        f.write(hdr)
        f.write("-" * len(hdr) + "\n")
        for name, M, m in model_rows:
            f.write(f"{name:<23}{M:>4}{fit_of(m):>20.2f}"
                    f"{m['lib']:>16.2f}{m['assign_cost']:>14.2f}{m['total']:>14.2f}\n")

        if mFull is not None:
            f.write(f" {'Per-cluster (full lib)':<23}{mFull['M']:>4}{fit_of(mFull):>20.2f}"
                    f"{mFull['lib']:>16.2f}{mFull['assign_cost']:>14.2f}"
                    f"{mFull['total']:>14.2f}\n")

    # --- console report ---
    print("\n================ Model comparison (one consistent cost model) ================")
    with open(table_path) as f:
        print(f.read())
    winner = min(((n, m["total"]) for n, _, m in model_rows), key=lambda t: t[1])
    # print(f"Lowest total L(D) [same-source models]: {winner[0]}  ({winner[1]:.2f} nats)")
    # if mFull is not None:
    #     print(f"Per-cluster (full lib), separate source, total = {mFull['total']:.2f} "
    #           f"nats -- compare by TOTAL only.")
    # print("(Ranked on TOTAL, not fit -- fewer functions => worse fit but lower "
    #       "function+assignment cost.)\n")

    if m2 is not None:
        print("M=2 best pair (both functions used, n>=1 each):")
        print(f"  A = {func_str[m2['func_a']]}   (n={m2['n_a']})")
        print(f"  B = {func_str[m2['func_b']]}   (n={m2['n_b']})")
        print(f"  global={m2['total']:.2f}  vs  greedy={m2['total_greedy']:.2f}")

        # Which clusters sit on the MINORITY side (usually the ones "pulling"
        # toward the second function), and by how much they prefer it. gap =
        # L_a - L_b: gap > 0 -> cluster fits B better -> it's the reason B exists.
        to_a, keep, gap = m2["to_a"], m2["keep"], m2["gap"]
        on_a = np.where(to_a & keep)[0]
        on_b = np.where((~to_a) & keep)[0]
        minority, majority = (on_b, on_a) if len(on_b) <= len(on_a) else (on_a, on_b)
        min_side = "B" if len(on_b) <= len(on_a) else "A"
        print(f"\n  Clusters on the minority side ({min_side}, {len(minority)} of "
              f"{len(on_a) + len(on_b)}) - these are what make a 2nd function pay off:")
        # sort minority by how strongly they prefer their side (largest |gap| first)
        order = sorted(minority, key=lambda i: -abs(gap[i]))
        for i in order:
            pref = "B" if gap[i] > 0 else "A"
            print(f"    {clusters[i]:<22} gap(L_A-L_B) = {gap[i]:+8.3f}  "
                  f"(fits {pref} better by {abs(gap[i]):.3f} nats)")
        # also show the closest majority-side cluster: the next one that would flip
        if len(majority):
            near = min(majority, key=lambda i: abs(gap[i]))
            print(f"  Closest cluster still on the majority side (nearly flips): "
                  f"{clusters[near]}  gap = {gap[near]:+.3f}")
    else:
        print("M=2: no pair admits a genuine two-function split (every pair "
              "collapses to one function). The universal M=1 model already "
              "captures this.")

    if m2g is not None:
        print(f"\nM=2 best GREEDY pair (each cluster picks its better of the two, "
              f"costs added after):")
        print(f"  A = {func_str[m2g['func_a']]}   (n={m2g['n_a']})")
        print(f"  B = {func_str[m2g['func_b']]}   (n={m2g['n_b']})")
        print(f"  greedy total={m2g['total']:.2f}"
              + (f"  vs  global best pair={m2['total']:.2f}" if m2 is not None else ""))

    print(f"\nPer-cluster global (top-{len(valid)} pool): M={mN['M']} functions used, "
          f"total={mN['total']:.2f} (greedy per-cluster would give "
          f"{mN['greedy_total']:.2f} using M={mN['greedy_M']}).")
    used = np.where(mN["counts"] > 0)[0]
    for u in used:
        print(f"    [{ids[u]:>3}] n={mN['counts'][u]:>3}  {func_str[u]}")

    # Where each coordinate-descent restart landed (the local-minimum landscape).
    if args.restart_log and "restart_log" in mN:
        logs = mN["restart_log"]
        n_distinct = len({round(e["total"], 6) for e in logs})
        print(f"\n  Restart landscape ({len(logs)} starts, {n_distinct} distinct "
              f"minima).  '<' marks a start that improved the best-so-far:")
        print(f"    {'start':<16}{'M':>4}{'total DL':>14}   split (func_id:n)")
        for e in logs:
            used_e = np.where(e["counts"] > 0)[0]
            split = " ".join(f"{ids[u]}:{e['counts'][u]}" for u in used_e)
            mark = " <" if e["is_best"] else "  "
            print(f"    {e['kind']:<16}{e['M']:>4}{e['total']:>14.2f}{mark} {split}")

    # Full-library per-cluster: each cluster's single best function over the WHOLE
    # library (from its final_*.dat), printed per cluster.
    if mFull is not None:
        print(f"\nPer-cluster FULL-LIBRARY: each cluster's best function over the "
              f"whole library (base run + best-funcs re-run merged), "
              f"total={mFull['total']:.2f} using M={mFull['M']} distinct functions "
              f"over {mFull['N']} clusters "
              f"({mFull['n_from_rerun']} took the improved re-run fit).")
        if mFull["missing"]:
            print(f"    [warn] {len(mFull['missing'])} cluster(s) had no readable "
                  f"final_*.dat and were skipped: {', '.join(mFull['missing'][:8])}"
                  + (" ..." if len(mFull["missing"]) > 8 else ""))
        print("    Best function per cluster (src = base | rerun):")
        for name in sorted(mFull["chosen"]):
            func, negll, cod, src = mFull["chosen"][name]
            print(f"      {name:<22} [{src:<5}] {func:<40} "
                  f"negll={negll:.2f} codelen={cod:.2f}")
        print("    Function usage counts:")
        for func, c in sorted(mFull["counts"].items(), key=lambda kv: -kv[1]):
            print(f"      n={c:>3}  {func}")

    saved = [str(table_path), str(out_pairs)]
    if greedy_pair_rows:
        saved.append(str(out_pairs_greedy))
    print("\nSaved: " + "\n       ".join(saved))


if __name__ == "__main__":
    main()
