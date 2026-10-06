"""Corrected ExcessSurfaceDensity integration, replacing esr/esd.py's ExcessSurfaceDensity.

Fixes three distinct, independently-verified numerical failure modes found by auditing
results_fixed_match/ (see notes.md for the full investigation and the history of grid
designs tried, including two -- a near-zero/near-R seamed grid and a cosine/Chebyshev
grid -- that were tried and REJECTED because they fixed one problem while quietly
breaking another; see "the seam lesson" below):

1. SMALL-r COLLAPSE. Functions like pow(Abs(a0), pow(Abs(a1), x)) can make the density
   profile collapse from >1e15 to O(1) within <2% of the radial range once a1 is small
   enough. The original first-term grid (np.linspace(MIN_INTEGRATION_RADIUS, R, N),
   evenly spaced in r) puts almost none of its N points inside that collapse -- the
   optimiser was free to make it arbitrarily narrow and get an arbitrarily wrong (and
   falsely GOOD) integral. Verified: at num_points=120, one such function's reported
   negloglike=80.53; the true value (adaptive quadrature) is >3.4e8. FIX: grid the
   first-term integral uniformly in sqrt(r) instead of r -- this concentrates points
   near r=0 while keeping full coverage out to R.

2. LARGE-r UNDER-RESOLUTION (second term) and NEAR-DATA-POINT UNDER-RESOLUTION (first
   term, near its own R endpoint) are the SAME disease as (1), just showing up at the
   other end of the domain: sqrt spacing buys density near r=0 by spending less near
   the far edge of whatever range it covers, and the far edge matters too (a narrow
   feature parked right at R, one of the actual data radii, changes the prediction and
   is exactly where an optimiser would want to hide one). THE SEAM LESSON: the first
   attempt to fix this added a *second*, separately-spaced sub-grid near R (or, in a
   second attempt, used a single cosine/Chebyshev-spaced grid dense at both ends).
   Both fixed the near-R bump problem -- and both silently made a plain, physically
   ordinary steep cusp (rho~a0/r**3) dramatically WORSE at large R (54% and 104% error
   respectively, vs ~5-60% for plain sqrt depending on R -- see notes.md for the full
   numbers). Root cause: any grid with an uneven point-density profile (a seam, or a
   deliberately sparse middle) can disrupt the trapezoid rule's usual partial error
   cancellation between neighbouring intervals for a SMOOTH, log-uniformly-spread
   integrand -- this is a real, verified phenomenon (see the "combining grids" numbers
   in notes.md), not a hypothetical one. FIX ACTUALLY ADOPTED: no seam, no second
   shape -- just plain uniform-in-sqrt(r) spacing (same as (1)) at a higher default
   num_points. Cost is negligible either way (JAX dispatch overhead dominates at these
   array sizes: 0.82ms/eval at N=500 vs 0.96ms/eval at N=5000 for a 10-radius array),
   so there is no real reason not to just use more points instead of a cleverer shape.
   Verified: rho~a0/r**3 at R=3.685 (the worst case found) goes from 60% error at
   N=500 to 0.02% at N=30000; N=5000 (the default here) gets both this case (~1%) and
   a narrow synthetic bump parked right at a data radius (~3.5% at bump-width=0.0003,
   vs 22-70% for the seamed/cosine alternatives at the same width) to a good place
   simultaneously, with no seam-related regression on anything else tested.
   Residual risk: an adversarially narrow-enough feature ANYWHERE can still in
   principle evade any finite N -- this raises the bar a great deal, it does not
   remove the bar. See notes.md for the general argument (no fixed, non-adaptive grid
   can be proven complete against an actively-searching optimiser).

3. POLE-AT-INTEGRATION-BOUNDARY EXPLOIT. Functions with a genuine pole (e.g.
   1/(a0+x)) can have their pole pulled to sit an ULP-scale distance from
   MIN_INTEGRATION_RADIUS (confirmed down to 1 part in 1e16-1e17 -- right at
   float64's precision limit). This is NOT a discretisation error: the resulting
   negloglike is a real, smooth, reproducible local minimum (verified by direct fine
   scan), but it's exploiting the trapezoid rule's fixed, non-infinitesimal weighting
   of its own boundary point to inject an artificial, freely-tunable rescaling of the
   ENTIRE predicted profile -- nothing to do with the genuine mathematical content of
   the function, and nothing to do with any actual observed data point (the pole
   sits ~4e5x closer to the origin than the nearest real measurement). No amount of
   quadrature refinement fixes this (a pole is genuinely infinite there); it needs to
   be excluded by construction. FIX: before integrating, probe the density at
   MIN_INTEGRATION_RADIUS and at MARGIN*MIN_INTEGRATION_RADIUS (MARGIN=100, i.e. still
   a wildly non-physical radius, 1e-4 by default). Their ratio is 1 for a flat
   profile and MARGIN**p for a genuine power-law cusp rho~r^-p; even an extremely
   steep (unphysical) p=3 cusp gives ratio~1e6. A ratio above POLE_GUARD_MAX_RATIO
   (default 1e8, i.e. requiring an effective local power-law index >4 right at the
   edge of the integration domain) is not a physically credible cusp -- it's this
   exploit -- and the whole evaluation is flagged invalid (returns inf, same
   contract as the existing check_density() invalid-density path). NOTE: this guard's
   two fixed probe points (RMIN, 100*RMIN) do NOT catch general small-r collapses
   located further out (verified blind spot) -- it is not a substitute for (1)/(2),
   only a complement, for the specific case of a literal, irreducible mathematical
   pole that no amount of resolution can fix.

Drop-in replacement: same call signature as esr.esd.ExcessSurfaceDensity.calculate.
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as np
import numpy
from dataclasses import dataclass

MIN_INTEGRATION_RADIUS = 1e-6
MAX_INTEGRATION_RADIUS = 1e4

POLE_GUARD_MARGIN = 100.0        # probe radius = MARGIN * MIN_INTEGRATION_RADIUS
POLE_GUARD_MAX_RATIO = 1e8       # flag if rho(r_min)/rho(margin*r_min) exceeds this

DEFAULT_NUM_POINTS = 5000        # see item 2 above -- cost is ~flat vs N=500, so just
                                  # use enough points rather than a cleverer grid shape


def atleast_kd(array, k, append_dims=True):
    array = np.asarray(array)
    if append_dims:
        new_shape = array.shape + (1,) * (k - array.ndim)
    else:
        new_shape = (1,) * (k - array.ndim) + array.shape
    return array.reshape(new_shape)


def trapz_(arr, axis, dx=None):
    arr = np.moveaxis(arr, axis, 0)
    if dx is None:
        dx = np.ones(arr.shape[0])
    if dx.ndim > 1:
        dx = np.moveaxis(dx, axis, 0)
    dx = atleast_kd(dx, arr.ndim)
    arr = dx * arr
    return 0.5 * (arr[0, ...] + 2 * arr[1:-1, ...].sum(axis=0) + arr[-1, ...])


@dataclass
class ExcessSurfaceDensityFixed:
    radii: np.ndarray
    density_func: object
    num_points: int
    radial_axis_to_broadcast: object
    density_axis: object
    params: tuple
    use_clipping: bool = False

    def _pole_guard_ok(self):
        """Reject profiles whose steepness at the integration floor is not
        explicable by any credible power-law cusp -- see module docstring, item 3."""
        r_lo = np.asarray(MIN_INTEGRATION_RADIUS)
        r_hi = np.asarray(MIN_INTEGRATION_RADIUS * POLE_GUARD_MARGIN)
        rho_lo = np.abs(self.density_func(r_lo, *self.params))
        rho_hi = np.abs(self.density_func(r_hi, *self.params))
        ratio = rho_lo / np.maximum(rho_hi, 1e-300)
        bad = np.logical_or(
            ~np.isfinite(ratio),
            ratio > POLE_GUARD_MAX_RATIO,
        )
        return ~bad

    def check_density(self, rhos):
        below_threshold = np.any(rhos < -1e-8)
        has_infinite_values = np.any(np.isinf(rhos))
        has_nan_values = np.any(np.isnan(rhos))
        return np.logical_or(np.logical_or(below_threshold, has_infinite_values), has_nan_values)

    def _monotonicity_ok(self, rhos_first):
        """Reject profiles whose density increases anywhere as radius grows -- see
        module docstring, item 4. Real density profiles decline monotonically
        outward; an increase means a hidden bump or pole is sitting somewhere away
        from the origin (e.g. a mid-domain pole that never flips sign, so it trips
        neither the negative-density nor the infinite-value check -- verified blind
        spot, see notes.md). Free check: reuses the first-term density values
        already computed for the integral itself, no extra evaluations. MONOTONICITY_TOL
        allows for ordinary floating-point noise without flagging it as a real increase.
        """
        diffs = rhos_first[1:, ...] - rhos_first[:-1, ...]
        tol = MONOTONICITY_TOL * np.abs(rhos_first[:-1, ...]) + 1e-300
        violated = np.any(diffs > tol)
        return ~violated

    def _first_term(self):
        """(4/R^2) * integral_{RMIN}^{R} rho(r) r^2 dr, gridded uniformly in sqrt(r)
        (dense near r=0). No seam, no second grid shape -- see item 2 in the module
        docstring for why a seamed/near-R-dense grid was tried and rejected. Accuracy
        near R (including for narrow features parked at the evaluation point) comes
        from using enough points (DEFAULT_NUM_POINTS), not from a cleverer shape.
        """
        R = self.radii
        frac = np.linspace(0.0, 1.0, self.num_points)[:, None]
        sqrt_rmin = np.sqrt(MIN_INTEGRATION_RADIUS)
        xs = (sqrt_rmin + frac * (np.sqrt(R[None, :]) - sqrt_rmin)) ** 2   # shape (N, nR)

        if self.use_clipping:
            xs_arg = np.clip(xs, a_min=None, a_max=3.0)
        else:
            xs_arg = xs
        rhos = self.density_func(xs_arg, *self.params)
        rhos = np.atleast_1d(rhos)

        integrand = 4 * rhos * xs ** 2
        dxs = atleast_kd(np.gradient(xs, axis=0), integrand.ndim)
        first_term = trapz_(integrand, axis=0, dx=dxs) / R ** 2
        return first_term, rhos

    def _second_term(self):
        """2 * integral_R^inf rho(r) r / sqrt(r^2-R^2) dr, via r=R/cos(theta),
        gridded uniformly in sqrt(pi/2 - theta) (dense near theta=pi/2, i.e. r->inf)."""
        R = self.radii
        VMIN = 1e-10
        frac = np.linspace(0.0, 1.0, self.num_points)
        sqrt_vmin = np.sqrt(VMIN)
        v = (sqrt_vmin + frac * (np.sqrt(numpy.pi / 2) - sqrt_vmin)) ** 2
        thetas = np.sort(numpy.pi / 2 - v)                      # ascending, dense near pi/2
        thetas_ = atleast_kd(thetas[:, None], self.radii.ndim + 1)

        cos_thetas_abs = np.abs(np.cos(thetas_))
        density_arg = R[None, ...] / cos_thetas_abs
        if self.use_clipping:
            density_arg = np.clip(density_arg, a_min=None, a_max=3.0)
        rhos = self.density_func(density_arg, *self.params)
        rhos = np.atleast_1d(rhos)

        radii_b = atleast_kd(R[None, ...], rhos.ndim)
        thetas_b = atleast_kd(thetas_, rhos.ndim)
        integrand = 4 * radii_b * rhos / (4 * np.sin(thetas_b) + 3 - np.cos(2 * thetas_b))
        dthetas = atleast_kd(np.gradient(thetas, axis=0), integrand.ndim)
        second_term = trapz_(integrand, axis=0, dx=dthetas)
        return second_term, rhos

    def esd(self):
        pole_ok = self._pole_guard_ok()

        first_term, rhos_first = self._first_term()
        second_term, rhos_second = self._second_term()

        has_invalid = np.logical_or(self.check_density(rhos_first), self.check_density(rhos_second))
        invalid = np.logical_or(has_invalid, ~pole_ok)

        result = np.where(invalid, np.full_like(self.radii, np.inf), first_term - second_term)
        return result

    @classmethod
    def calculate(cls, radii, density_func, num_points=DEFAULT_NUM_POINTS, radial_axis_to_broadcast=None,
                  density_axis=None, params=(), use_cliping=False):
        return cls(
            radii=radii,
            density_func=density_func,
            num_points=num_points,
            radial_axis_to_broadcast=radial_axis_to_broadcast,
            density_axis=density_axis,
            params=params,
            use_clipping=use_cliping,
        ).esd()


# Drop-in alias matching the original class name, for minimal-diff swapping into
# WL_likelihood_CLASH.py / WL_likelihood.py (`from esd_fixed import ExcessSurfaceDensity`).
ExcessSurfaceDensity = ExcessSurfaceDensityFixed
