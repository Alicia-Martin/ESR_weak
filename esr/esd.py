# import numpy as np
from dataclasses import dataclass
import jax
jax.config.update("jax_enable_x64", True)
jax.numpy.array(1, dtype=int)
import jax.numpy as np
import scipy.integrate as integrate
import sys
import jax
import matplotlib.pyplot as plt
import numpy
from jax import numpy as jnp
from scipy.special import roots_legendre
from quadax import quadgk



MIN_INTEGRATION_RADIUS = 1e-6
MAX_INTEGRATION_RADIUS = 1e4
MAX_ERROR_TOLERANCE = 10



def atleast_kd(array, k, append_dims=True):
    array = np.asarray(array)

    if append_dims:
        new_shape = array.shape + (1,) * (k-array.ndim)
    else:
        new_shape = (1,) * (k-array.ndim) + array.shape

    return array.reshape(new_shape)


def trapz_(arr, axis, dx=None):
    arr = np.moveaxis(arr, axis, 0)

    if dx is None:
        dx = np.ones(arr.shape[0])
    if dx.ndim > 1:
        dx = np.moveaxis(dx, axis, 0)

    dx = atleast_kd(dx, arr.ndim)

    arr = dx*arr

    return 0.5*(arr[0, ...] + 2*arr[1:-1, ...].sum(axis=0) + arr[-1, ...])

def cumtrapz_(arr, axis, dx=None):
    """
    Calculate the cumulative trapezoidal integral along a specified axis.
    
    Parameters
    ----------
    arr : array_like
        Input array to integrate.
    axis : int
        Axis along which to integrate.
    dx : float or array_like, optional
        Spacing between elements along the integration axis. Default is 1.
        If array_like, it must be broadcastable to the shape of `arr` along `axis`.

    Returns
    -------
    cumulative_integral : ndarray
        The result of the cumulative integration. Its shape is the same as `arr`.
    """
    original_ndim = arr.ndim
    arr_moved = np.moveaxis(arr, axis, 0) # Move integration axis to the first position

    if dx is None:
        # Default dx is 1 for each step
        dx_val = np.ones(arr_moved.shape[0] - 1) # dx applies to intervals between points
    elif np.isscalar(dx):
        dx_val = np.full(arr_moved.shape[0] - 1, dx)
    else: # dx is an array
        if dx.ndim != 1 or dx.shape[0] != arr_moved.shape[0] - 1:
            # Handle cases where dx might need to be broadcast or aligned
            # For simplicity, assume dx is 1D and correct length or scalar
            dx_val = np.array(dx)
            if dx_val.shape[0] != arr_moved.shape[0] - 1:
                raise ValueError("dx must be a scalar or an array of length arr.shape[axis]-1")
        dx_val = dx

    # Ensure dx_val is broadcastable correctly, especially if arr_moved is >1D
    # It needs to be (N_intervals, 1, 1, ...) for proper broadcasting with arr_moved[:-1] + arr_moved[1:]
    dx_val = atleast_kd(dx_val, arr_moved.ndim) # Adds necessary trailing dimensions

    # Calculate trapezoidal areas for each segment
    # 0.5 * (f(x_i) + f(x_{i+1})) * dx_i
    areas = 0.5 * (arr_moved[:-1] + arr_moved[1:]) * dx_val

    # Perform cumulative sum of these areas
    cumulative_integral_moved = np.cumsum(areas, axis=0)
    zeros_shape = (1,) + arr_moved.shape[1:]
    cumulative_integral_moved = np.concatenate((np.zeros(zeros_shape, dtype=arr_moved.dtype), cumulative_integral_moved), axis=0)

    # Move axis back to original position
    cumulative_integral = np.moveaxis(cumulative_integral_moved, 0, axis)

    return cumulative_integral



class LargeQuadratureErrorsException(Exception):
    pass


def _check_errors_ok(error):
    try:
        max_error = error.max()
    except AttributeError:
        max_error = error
    if max_error > MAX_ERROR_TOLERANCE:
        raise LargeQuadratureErrorsException(
            f'Maximum quadrature error ({round(max_error, 2)}) is very large and indicates a problem',
        )


@dataclass
class SurfaceDensity:
    radii: np.ndarray
    density_func: object
    num_points: int
    radial_axis_to_broadcast: object
    density_axis: object
    params: tuple  # Add params here

    def _pad_radii_and_thetas_for_argument(self, thetas, radii):
        radii_ = radii.reshape(radii.shape + thetas.ndim*(1,))
        thetas_ = thetas.reshape(radii.ndim*(1,) + thetas.shape)
        if self.radial_axis_to_broadcast is not None:
            radii_ = np.moveaxis(radii_, self.radial_axis_to_broadcast, -1)
            thetas_ = np.moveaxis(thetas_, self.radial_axis_to_broadcast, -1)
        return thetas_, radii_

    def _pad_radii_and_thetas_for_integrand(self, thetas, radii, rhos):
        radii_ = radii.reshape(radii.shape + thetas.ndim*(1,))
        thetas_ = thetas.reshape(radii.ndim*(1,) + thetas.shape)
        radii__ = atleast_kd(radii_, rhos.ndim)
        thetas__ = atleast_kd(thetas_, rhos.ndim)
        if self.radial_axis_to_broadcast is not None:
            radii__ = np.moveaxis(radii__, self.radial_axis_to_broadcast, self.density_axis)
            thetas__ = np.moveaxis(thetas__, self.radial_axis_to_broadcast, self.density_axis)
        return thetas__, radii__

    def _sd_integrand_func(self, thetas):
        thetas_, radii_ = self._pad_radii_and_thetas_for_argument(thetas, self.radii)
        density_arg = radii_/np.cos(thetas_)
        rhos = self.density_func(density_arg, *self.params)  # Pass params to density_func
        thetas__, radii__ = self._pad_radii_and_thetas_for_integrand(thetas, self.radii, rhos)
        return 2 * radii__ * rhos/(np.cos(thetas__)**2)

    def sd(self):
        thetas = np.linspace(0, np.pi/2, self.num_points)
        dthetas = np.gradient(thetas, axis=0)
        integrand = self._sd_integrand_func(thetas)
        return trapz_(integrand, axis=1, dx=dthetas)

    @classmethod
    def calculate(cls, radii, density_func, num_points=120, radial_axis_to_broadcast=None, density_axis=-1, params=()):
        # print('num', num_points)
        return cls(
            radii=radii,
            density_func=density_func,
            num_points=num_points,
            radial_axis_to_broadcast=radial_axis_to_broadcast,
            density_axis=density_axis,
            params=params
        ).sd()


@dataclass
class SurfaceDensity2:
    radii: np.ndarray
    density_func: object
    num_points: int
    radial_axis_to_broadcast: object
    params: tuple  # Add params here

    def _pad_radii_and_ells_for_argument(self, ells, radii):
        radii_ = atleast_kd(radii, radii.ndim+ells.ndim)
        ells_ = atleast_kd(ells, radii.ndim+ells.ndim, append_dims=False)
        if self.radial_axis_to_broadcast is not None:
            radii_ = np.moveaxis(radii_, self.radial_axis_to_broadcast, radii_.ndim-1)
            ells_ = np.moveaxis(ells_, self.radial_axis_to_broadcast, ells_.ndim-1)
        return ells_, radii_

    def _sd_alt_integrand_func(self, ells):
        ells_, radii_ = self._pad_radii_and_ells_for_argument(ells, self.radii)
        density_arg = np.sqrt(radii_**2 + ells_**2)
        rhos = self.density_func(density_arg, *self.params)  # Pass params to density_func
        return 2*rhos

    def sd(self):
        ells = np.geomspace(MIN_INTEGRATION_RADIUS, MAX_INTEGRATION_RADIUS, self.num_points)
        d_ells = np.gradient(ells, axis=0)
        integrand = self._sd_alt_integrand_func(ells)
        return trapz_(integrand, axis=1, dx=d_ells)

    @classmethod
    def calculate(cls, radii, density_func, num_points=120, radial_axis_to_broadcast=None, density_axis=None, params=()):
        return cls(
            radii=radii,
            density_func=density_func,
            num_points=num_points,
            radial_axis_to_broadcast=radial_axis_to_broadcast,
            params=params
        ).sd()

import inspect
@dataclass
class ExcessSurfaceDensity:
    radii: np.ndarray
    density_func: object
    num_points: int
    radial_axis_to_broadcast: object
    density_axis: object
    params: tuple  # Store additional parameters
    use_clipping: bool = False  # Add use_log_mode parameter

    def check_density(self, rhos):
        # Check if any densities are below the threshold (-1e-8)
        below_threshold = np.any(rhos < -1e-8)
            
        # Check if any densities are infinite
        has_infinite_values = np.any(np.isinf(rhos))
        # jax.debug.print('below_threshold {x}', x = rhos)

        return np.logical_or(below_threshold, has_infinite_values)
        # return False
        # return below_threshold
    
    def _esd_first_term_integrand_func(self, xs):
        """First term integrand function for ESD calculation."""
        # jax.debug.print('xs {x}', x = xs)

        if getattr(self, "use_clipping", False):
            # print('Using clipping for xs')
            MAX_DENSITY_ARG_CLIP = 3.0 # Experiment with this value. Too low may cut off physics.
            xs = jnp.clip(xs, a_min=None, a_max=MAX_DENSITY_ARG_CLIP)

        rhos = self.density_func(xs, *self.params)
        rhos = np.atleast_1d(rhos)
        # jax.debug.print('params {x}', x = self.params)
        # jax.debug.print('xs {x}', x = xs)
        # jax.debug.print('rhos {x}', x = rhos)
        # postfactor = xs**2 / atleast_kd(self.radii, xs.ndim, append_dims=False)**2
        postfactor = xs**2

        postfactor = atleast_kd(postfactor, rhos.ndim)

        if self.radial_axis_to_broadcast is not None:
            postfactor = np.moveaxis(postfactor, self.radial_axis_to_broadcast + 1, self.density_axis)

        # print('HERE', 4 * rhos * postfactor)
        # print(len(rhos), len(postfactor))
        # jax.debug.print('postfactor {x}', x = 4 * rhos * postfactor)
        return 4 * rhos * postfactor, rhos
    
    # @profile
    def _esd_second_term_integrand_func(self, thetas):
        """Second term integrand function for ESD calculation."""
        thetas_ = atleast_kd(thetas, self.radii.ndim + 1)
        cos_thetas_abs = jnp.abs(jnp.cos(thetas_))
        density_arg = self.radii[None, ...] / cos_thetas_abs
        # jax.debug.print('density_arg {x}', x = density_arg)

        if getattr(self, "use_clipping", False):
            # jax.debug.print('Using clipping for density_arg')
            #density_func is the log of self.density_func
            # log_rhos = self.density_func(density_arg, *self.params)
            # jax.debug.print('log_rhos')
            MAX_DENSITY_ARG_CLIP = 3.0 # Experiment with this value. Too low may cut off physics.
            density_arg = jnp.clip(density_arg, a_min=None, a_max=MAX_DENSITY_ARG_CLIP)
            rhos = self.density_func(density_arg, *self.params)
            
            # rhos = jnp.exp(log_rhos)

        else:
            rhos = self.density_func(density_arg, *self.params)
        # jax.debug.print('rhos {x}', x = rhos)
        #check if there are negative density values
        

        #calculate gradient of rhos
        # grad_rhos = jnp.gradient(rhos, axis=0)
        # jax.debug.print('grad_rhos {x}', x = grad_rhos)
        #calculate the gradient with jax
        # arg = np.linspace(1,10,10)
        # rhos_try = self.density_func(arg, *self.params)
        # #clip rhos_try to avoid inf
        # rhos_try = jnp.clip(rhos_try, a_min=1e-8)
        # eps = 1e-8

        # def clipped_density_func(a):
        #     # Evaluate the raw density
        #     density = self.density_func(a, *self.params)
        #     # Avoid exact zero or negative values (principled smoothing)
        #     density_safe = jnp.maximum(density, eps)
        #     return density_safe

        # grad_rhos_jax = jax.vmap(lambda x: jax.grad(clipped_density_func)(x))(arg)
        # jax.debug.print('rhos_try {x}', x = rhos_try)
        # jax.debug.print('grad_rhos_jax {x}', x = grad_rhos_jax)
        
        rhos = np.atleast_1d(rhos)

        radii = atleast_kd(self.radii[None, ...], rhos.ndim)
        if self.radial_axis_to_broadcast is not None:
            radii = np.moveaxis(radii, self.radial_axis_to_broadcast + 1, self.density_axis)
        thetas__ = atleast_kd(thetas_, rhos.ndim)
        return 4 * radii * rhos / (4 * np.sin(thetas__) + 3 - np.cos(2 * thetas__)), rhos
        
        # return rhos

    def _esd_second_term_func(self):
        """Second term integrand using Gaussian quadrature for better stability."""

        # Get Gaussian quadrature nodes and weights in [0, π/2]
        thetas_x, weights = roots_legendre(self.num_points)
        thetas = 0.5 * (thetas_x + 1) * (np.pi / 2)
        weights = 0.5 * weights * (np.pi / 2)

        # Broadcast to match shape for vectorised evaluation
        thetas_broadcast = atleast_kd(thetas[:, None], self.radii.ndim + 1)
        cos_thetas = jnp.abs(jnp.cos(thetas_broadcast))
        density_args = self.radii[None, ...] / cos_thetas

        # if getattr(self, "use_clipping", False):
        # jax.debug.print('Using clipping for density_args')
        MAX_DENSITY_ARG_CLIP = 3
        density_args = jnp.clip(density_args, a_min=None, a_max=MAX_DENSITY_ARG_CLIP)

        rhos = self.density_func(density_args, *self.params)
        # jax.debug.print('rhos {x}', x = rhos)
        rhos = np.atleast_1d(rhos)

        # Prepare R and theta terms for integration
        R_broadcast = atleast_kd(self.radii[None, ...], rhos.ndim)
        thetas_broadcast = atleast_kd(thetas[:, None], rhos.ndim)

        integrand = R_broadcast * rhos / (4 * jnp.sin(thetas_broadcast) + 3 - jnp.cos(2 * thetas_broadcast))

        # Perform weighted sum along theta axis
        second_term = jnp.sum(weights[:, None] * integrand, axis=0)
        return second_term
    
    def _esd_second_term_grid_func(self):
        MAX_DENSITY_ARG = 1/np.cos(np.pi/2)*np.max(self.radii)
        radii = self.radii

        log_r_targets = np.logspace(np.log10(radii.min()), np.log10(MAX_DENSITY_ARG), self.num_points)
        log_r_targets_2d = log_r_targets[:, None]  # shape (num_points, 1)
        radii_2d = atleast_kd(radii[None, :], 2)   # shape (1, num_radii)

        # Compute cosθ = r / (r / cosθ), so θ = arccos(r / r_target)
        cos_thetas = radii_2d / log_r_targets_2d
        cos_thetas = np.clip(cos_thetas, -1.0, 1.0)  # Clip to avoid NaNs
        thetas = np.arccos(cos_thetas)

        # Compute corresponding r / cosθ values (same as log_r_targets)
        density_args = radii_2d / cos_thetas

        if self.use_clipping:
            MAX_CLIP = 10.0
            density_args = np.clip(density_args, a_min=None, a_max=MAX_CLIP)

        rhos = self.density_func(density_args, *self.params)
        rhos = np.atleast_1d(rhos)
        # jax.debug.print('rhos {x}', x = rhos)

        # Expand radii and thetas to match shape
        R_broadcast = atleast_kd(radii_2d, rhos.ndim)
        thetas_broadcast = atleast_kd(thetas, rhos.ndim)

        # Compute integrand
        integrand = R_broadcast * rhos / (4 * jnp.sin(thetas_broadcast) + 3 - jnp.cos(2 * thetas_broadcast))

        # Compute ∆θ for each row
        dthetas = np.gradient(thetas, axis=0)

        # Integrate along θ axis
        second_term = trapz_(integrand, axis=0, dx=dthetas)

        return second_term


    # @profile
    def esd(self):
        # r_grid = np.linspace(MIN_INTEGRATION_RADIUS, np.max(self.radii), self.num_points*10)
        # #append self.radii to the grid
        # r_grid = np.concatenate((r_grid, self.radii))
        # #order the grid
        # r_grid = np.sort(r_grid)
        # #get inidces of self.radii in r_grid
        # indices = np.searchsorted(r_grid, self.radii)

        xs = np.linspace(MIN_INTEGRATION_RADIUS, self.radii, self.num_points)
        #do xs in log space
        # xs = np.logspace(np.log10(MIN_INTEGRATION_RADIUS), np.log10(np.max(self.radii)), self.num_points)
        # print('xs', xs)
        # xs = xs.reshape((xs.size, 1))  # Ensure xs is a column vector for broadcasting

        first_term_integrand, rhos_first_term = self._esd_first_term_integrand_func(xs)
        # plt.plot(xs, first_term_integrand, label='First term integrand')
        # plt.show()
        # sys.exit()
        # jax.debug.print('first_term_integrand {x}', x = first_term_integrand)
        # jax.debug.print('rhos_first_term {x}', x = rhos_first_term)
        # first_term_integrand, rhos_first_term = self._esd_first_term_integrand_func(r_grid)

        # Check if there are negative density values
        has_negative_rhos = self.check_density(rhos_first_term)
        # jax.debug.print('density {x}', x = has_negative_rhos)
        # has_negative_rhos = False

        thetas = np.linspace(0, np.pi / 2, self.num_points)
        dthetas = np.gradient(thetas, axis=0)
        second_term_integrand, rhos_second = self._esd_second_term_integrand_func(thetas)
        # jax.debug.print('second_term_integrand {x}', x = rhos_second)

        has_negative_rhos_second_term = self.check_density(rhos_second)
        #combine with previous has_negative_rhos
        has_negative_rhos = np.logical_or(has_negative_rhos, has_negative_rhos_second_term)
        # jax.debug.print('has_negative_rhos {x}', x = has_negative_rhos)
        # has_negative_rhos = False

        # @profile
        def compute_valid_esd():
            dxs = atleast_kd(np.gradient(xs, axis=0), first_term_integrand.ndim)
            # jax.debug.print('dxs {x}', x = dxs)
            # dxs = r_grid[1:] - r_grid[:-1]
            # jax.debug.print('first_term_integrand {x}', x=first_term_integrand)
            # first_term = cumtrapz_(first_term_integrand, axis=0, dx=dxs)
            # first_term = first_term[indices]/self.radii**2
            first_term = trapz_(first_term_integrand, axis=0, dx=dxs)
            first_term = first_term/self.radii**2
            # jax.debug.print('first_term {x}', x = first_term)
            # jax.debug.print('dx {x}', x = dxs)
            if self.radial_axis_to_broadcast is not None:
                dxs = np.moveaxis(dxs, self.radial_axis_to_broadcast + 1, self.density_axis)

            # NORMAL
            second_term = trapz_(second_term_integrand, axis=0, dx=dthetas)

            #update has_negative_rhos

            #GRID
            # second_term = self._esd_second_term_grid_func()

            #QUAD
            def integrand(theta):
                return self._esd_second_term_integrand_func(theta)
            # second_term, info = quadgk(integrand, [thetas[0], thetas[-1]])

            # GAUSSIAN QUAD
            # second_term = self._esd_second_term_func()
            # jax.debug.print('first_term {x}', x = first_term)
            # jax.debug.print('second_term {x}', x = second_term)
            # jax.debug.print('esd {x}', x = first_term - second_term)

            return first_term - second_term
            # return first_term
            # return second_term

        def handle_invalid_density():
            return np.full_like(self.radii, np.inf)
        
        # jax.debug.print('has_negative_rhos {x}', x = has_negative_rhos)

        esd_result = jax.numpy.where(
            has_negative_rhos,
            handle_invalid_density(),
            compute_valid_esd(),
        )
        # jax.debug.print('{x}', x = esd_result)
        return esd_result

    @classmethod
    def calculate(cls, radii, density_func, num_points=120, radial_axis_to_broadcast=None, density_axis=None, params=(), use_cliping=False):

        return cls(
            radii=radii,
            density_func=density_func,
            num_points=num_points,
            radial_axis_to_broadcast=radial_axis_to_broadcast,
            density_axis=density_axis,
            params=params,
            use_clipping = use_cliping
        ).esd()


@dataclass
class QuadExcessSurfaceDensity:
    radii: np.ndarray
    density_func: object
    num_points: int
    radial_axis_to_broadcast: object
    density_axis: object
    params: tuple

    def _esd_first_term_integrand_func(self, xs, radius):
        rhos = self.density_func(xs, *self.params)
        postfactor = xs**2 / atleast_kd(radius, xs.ndim, append_dims=False)**2
        postfactor = atleast_kd(postfactor, rhos.ndim)
        return 4 * rhos * postfactor

    def _esd_second_term_integrand_func(self, thetas):
        thetas_ = atleast_kd(thetas, self.radii.ndim + 1)
        density_arg = self.radii[None, ...] / np.cos(thetas_)
        rhos = self.density_func(density_arg, *self.params)
        radii = atleast_kd(self.radii[None, ...], rhos.ndim)
        thetas = atleast_kd(thetas_, rhos.ndim)
        return 4 * radii * rhos / (4 * np.sin(thetas) + 3 - np.cos(2 * thetas))

    def _integrate_esd_first_term_quad(self):
        rflats = self.radii.flatten()
        first_term_integral = numpy.array(
            [integrate.quad_vec(
                lambda x: self._esd_first_term_integrand_func(np.array([x]), rflats[i:i+1]),
                MIN_INTEGRATION_RADIUS,
                rflats[i],
            ) for i in range(self.radii.size)],
            dtype=object,
        )
        _check_errors_ok(first_term_integral[:, 1].astype(float))
        return numpy.concatenate(first_term_integral[:, 0]).astype(float)

    def _integrate_esd_second_term_quad(self):
        second_term, second_term_errors = integrate.quad_vec(
            lambda theta: self._esd_second_term_integrand_func(np.array([theta])),
            0,
            np.pi / 2,
        )
        _check_errors_ok(second_term_errors)
        return second_term

    def esd(self):
        dens_shape = self.density_func(self.radii, *self.params).shape
        first_term = self._integrate_esd_first_term_quad().reshape(dens_shape)
        second_term = self._integrate_esd_second_term_quad().reshape(dens_shape)
        return first_term - second_term

    @classmethod
    def calculate(cls, radii, density_func, num_points=120, radial_axis_to_broadcast=None, density_axis=None, params=()):
        return cls(
            radii=radii,
            density_func=density_func,
            num_points=num_points,
            radial_axis_to_broadcast=radial_axis_to_broadcast,
            density_axis=density_axis,
            params=params
        ).esd()
    

# if __name__=="__main__":

#     def myrho(r, a):
#         return 1/r
#     radii = np.linspace(0.1, 100, 100)
#     a = []
#     rho_func = myrho
#     esds = ExcessSurfaceDensity.calculate(radii, rho_func, params=(a,))

#     print(esds)

