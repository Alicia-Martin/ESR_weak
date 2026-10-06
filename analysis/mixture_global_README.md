# Globally-optimal mixture assignment (`mixture_global.py`)

Rebuilds the profile→cluster assignment analysis (paper §5.4 / Table 4) for
**CLASH**, minimising the *total* description length instead of choosing greedily.
Emits all three models under one consistent cost. Keeps `all_two_models_2.py`
and `different_models_2.py` untouched for reproducibility.

## The models it computes (Table 4, one cost model)

| Model | Functions | Solver | Certified optimum? |
|---|---|---|---|
| Universal (M=1) | 1 shared | pick the single best column, `L_assign = 0` | yes |
| Two profiles (global) | 2 | exact threshold sweep over all pairs | **yes** |
| Two profiles (greedy) | 2 | each cluster picks its better of the two (old `all_two_models_2.py`) | it's the baseline |
| Per-cluster (global) | any of the top-N | coordinate descent over the resolved pool | no (strong heuristic) |
| Per-cluster (greedy) | any of the top-N | each cluster → its single-best in the pool | it's the baseline |
| Per-cluster (full lib) | any in the WHOLE library | each cluster's best function read from its own `final_*.dat` | data-defined |

All use the **entropy** assignment cost (paper Eq. 34),
`L_assign = -Σ n_m log(n_m/N)`, so the rows of Table 4 are comparable.
(`different_models_2.py` defaulted to a *different* enumerative cost + mixture
penalty — that's the inconsistency this script removes.)

**Two profiles (greedy)** and **Per-cluster (full lib)** reproduce your two older
analyses inside this one consistent framework:
- *greedy pair* = the old `all_two_models_2.py` local choice (each cluster picks
  its better of the two functions, costs added after). Written to
  `two_func_pairs_greedy.txt`.
- *full library* = the old `different_models_2.py` idea, but each cluster's best
  function is taken from the **entire** library (its own `final_*.dat`), not the
  top-N pool. The console prints the chosen function **per cluster** plus usage
  counts.

## What changed vs. the old scripts

**vs. `all_two_models_2.py` (M=2).** The old script assigns each cluster
greedily to `argmin(negloglike + codelen)`, then adds `L_assign` **after** the
split is frozen — a *local* choice. Because `L_assign` depends only on the
counts, a near-tie cluster can be sent the wrong way. Here, M=2 is solved
**exactly**: sort clusters by the fit-gap `delta_i = L_a(i) - L_b(i)`, sweep the
threshold; `L_assign` is constant on each of the `N+1` segments, so each split is
O(1) from prefix sums → exact optimum in `O(N log N)` per pair. Greedy is the
single threshold at `delta = 0`, so `Total_global <= Total_greedy` always.

**vs. `different_models_2.py` (per-cluster / §5.4.1).** The old per-cluster row
also chose greedily (each cluster → its own single-best fit) with structure +
assignment bolted on after — the same flaw, and it's the model where it hurts
most (it fragments into ~51 functions because nothing penalises fragmentation
during the choice). Here the per-cluster model is coordinate descent over the
whole candidate pool minimising the **total** DL, so each cluster's choice finally
accounts for library + assignment cost. Reported alongside the old greedy total.

**Single source: the best-funcs run.** The old HSC per-complexity `NORMAL/BEST`
switch (comp 7,8 → `best_funcs`/`_600`) is gone. We only rank the top-20
functions, and those all live in the best-funcs library (the merged/snapped
re-run, more optimiser iterations → better convergence), so there is one library
+ one fit run:

- **library:** `function_library/best_funcs_change_snapping`
- **fits:** `output_WL_<cluster>_best_funcs_change_snapping_merged`

(If you ever widen the pool to functions only in `core_maths`, add a `"main"`
entry to `SOURCES` and put it in `SOURCE_ORDER` — the fallback machinery is still
there.)

Index lookup follows `build_full_clash_ranking.py` (the authoritative reader):
a function string → its unique index `u` in `unique_equations_{c}.txt`, then the
best `(negloglike+codelen)` over all all-equations rows `i` with
`matches_{c}.txt[i] == u` in `codelen_matches_comp{c}.dat`.

## Run it (on Glamdring, where the fit files live)

```bash
cd .../DM_esr/CLASH/ESR    # wherever the data + esr/fitting/output live
python mixture_global.py                       # Table 4 (best-funcs run)
python mixture_global.py --assignment enumerative   # old Option A cost (to reproduce)
python mixture_global.py --restarts 20         # more restarts for per-cluster CD
```

Outputs, in `esr/fitting/output/combining_clusters/`:

- `table4_global.txt` — the full model summary (Fit / Function cost / Assignment
  / Total) for every model above. Drop-in for §5.4.
- `two_func_pairs_global.txt` — **every** pair (global sweep), best first, with
  all terms broken out like the old `all_two_models_2.py`: `NegL`, `CodeL`,
  `NegL+CodeL`, `Lib`, `Assign`, `Total_DL`, the split counts `n_A`/`n_B`, and a
  final `least_used_func_ID: clusters` column naming the clusters assigned to the
  *less-populated* function of that pair (keyed by its function ID).
- `two_func_pairs_greedy.txt` — the same table for the **greedy** pair method
  (each cluster picks its better of the two), so you can compare global vs.
  greedy per pair.

The console prints the table, the winning model, the best global and greedy M=2
pairs, the top-N per-cluster breakdown, and — for the full-library model — the
**best function chosen for each cluster** plus function-usage counts.

## Diagnostics (read these if something looks off)

- `SymPy canonicalisation: ON/OFF` — printed at startup. If **OFF**, SymPy isn't
  installed in the job's environment, so reparametrised/simplified functions
  (`pow(Abs(a1),x)`, `x*(-a0+x)-x`, …) will NOT resolve. Install SymPy on the
  node (`pip install sympy`) — matching is identical to `build_full_clash_ranking.py`
  only when it's ON.
- Each skipped function prints **why**: `parse` (SymPy couldn't sympify it),
  `not_in_library` (canonical form absent for that complexity), `missing_lib`
  (that complexity's `unique_equations` file not found), `no_rows` (found but no
  `matches` rows point to it).
- `[info] finite L cells: mean=… min=… max=…` — per-(cluster,function) fit
  magnitude. The paper is ~5.5 nats/cluster; a wildly different mean flags a bad
  read (wrong `matches` rows, or `codelen_matches` not per-cluster).
- `[warn] L-matrix has K missing (inf) cells` — lists dead clusters and
  per-function missing counts. With the strict universal model, one missing cell
  in a function disqualifies it as a universal profile.

## Check the math without any data

```bash
python mixture_global.py --self-test
```

Runs locally, no files. Asserts: (1) `L_assign([62,87]) = 101.17` (paper Eq. 34);
(2) the M=2 sweep equals brute-force `2^N` and is never worse than greedy over
200 random cases; (3) a constructed case where global strictly beats greedy;
(4) coordinate descent beats its greedy init; (5) on a synthetic self-similar
sample, M=2 and per-cluster both collapse back to M=1 (the "universal wins"
behaviour).

## Paragraph for §5.4 (draft, edit to taste)

> We strengthen the mixture analysis by minimising the total mixed-model
> description length directly rather than letting each cluster choose greedily.
> Because the assignment cost of Eq. (34) depends only on the number of clusters
> assigned to each profile, the globally optimal two-profile split is found by
> ordering clusters by their difference in per-cluster description length and
> sweeping the partition threshold, which evaluates every candidate split exactly
> at O(N log N) cost; the greedy assignment is the special case in which the
> threshold sits where the two per-cluster costs are equal, and so can never give
> a lower total description length. We apply the same total-DL objective to the
> per-cluster model via coordinate descent over the full candidate set, so that
> each cluster's choice accounts for the structural and assignment costs it
> induces. [Report whether the global two-profile split lowers L(D) from the
> greedy value, whether the globally-optimised per-cluster L(D) falls below the
> greedy per-cluster value, and confirm that the universal single profile remains
> the preferred (lowest-L(D)) description of the sample.]
