# import numpy as np
from dataclasses import dataclass
import jax.numpy as np
import scipy.integrate as integrate
import sys
import jax
import matplotlib.pyplot as plt
import numpy


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


@dataclass
class ExcessSurfaceDensity:
    radii: np.ndarray
    density_func: object
    num_points: int
    radial_axis_to_broadcast: object
    density_axis: object
    params: tuple  # Store additional parameters

    def check_density(self, rhos):
        # Check if any densities are below the threshold (-1e-8)
        below_threshold = np.any(rhos < -1e-8)
        
        # Check if any densities are infinite
        # has_infinite_values = np.any(np.isinf(rhos))
        

        # return np.logical_or(below_threshold, has_infinite_values)
        return below_threshold
    
    def _esd_first_term_integrand_func(self, xs):
        """First term integrand function for ESD calculation."""
        rhos = self.density_func(xs, *self.params)
        # jax.debug.print('rho {x}', x = rhos)
        # jax.debug.print('xs {x}', x = xs)
        radii_ = self.radii
        postfactor = xs**2 / atleast_kd(self.radii, xs.ndim, append_dims=False)**2
        postfactor = atleast_kd(postfactor, rhos.ndim)
        if self.radial_axis_to_broadcast is not None:
            postfactor = np.moveaxis(postfactor, self.radial_axis_to_broadcast + 1, self.density_axis)
        return 4 * rhos * postfactor, rhos

    def _esd_second_term_integrand_func(self, thetas):
        """Second term integrand function for ESD calculation."""
        thetas_ = atleast_kd(thetas, self.radii.ndim + 1)
        density_arg = self.radii[None, ...] / np.abs(np.cos(thetas_))
        rhos = self.density_func(density_arg, *self.params)
        radii = atleast_kd(self.radii[None, ...], rhos.ndim)
        if self.radial_axis_to_broadcast is not None:
            radii = np.moveaxis(radii, self.radial_axis_to_broadcast + 1, self.density_axis)
        thetas__ = atleast_kd(thetas_, rhos.ndim)
        return 4 * radii * rhos / (4 * np.sin(thetas__) + 3 - np.cos(2 * thetas__))

    def esd(self):
        xs = np.linspace(MIN_INTEGRATION_RADIUS, self.radii, self.num_points)
        first_term_integrand, rhos_first_term = self._esd_first_term_integrand_func(xs)

        # Check if there are negative density values
        has_negative_rhos = self.check_density(rhos_first_term)

        def compute_valid_esd():
            dxs = atleast_kd(np.gradient(xs, axis=0), first_term_integrand.ndim)
            if self.radial_axis_to_broadcast is not None:
                dxs = np.moveaxis(dxs, self.radial_axis_to_broadcast + 1, self.density_axis)

            thetas = np.linspace(0, np.pi / 2, self.num_points)
            dthetas = np.gradient(thetas, axis=0)
            second_term_integrand = self._esd_second_term_integrand_func(thetas)

            first_term = trapz_(first_term_integrand, axis=0, dx=dxs)
            second_term = trapz_(second_term_integrand, axis=0, dx=dthetas)

            # num_error = np.sum((first_term - second_term)/first_term)
            # MAX_ERROR_TOLERANCE = 1e-3

            # def do_nothng():
            #     jax.debug.print('this one is fine')
            #     pass
            # def handle_large_error():
            #     jax.debug.print('WARNING: ESD calculation has a large numerical error {x}', x = self.params)

            # jax.lax.cond(num_error < MAX_ERROR_TOLERANCE, handle_large_error, do_nothng)
            # jax.debug.print('second {x}', x = first_term )
            return first_term - second_term

        def handle_invalid_density():
            return np.full_like(self.radii, np.inf)

        esd_result = jax.numpy.where(
            has_negative_rhos,
            handle_invalid_density(),
            compute_valid_esd(),
        )
        return esd_result

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
        first_term_integral = np.array(
            [integrate.quad_vec(
                lambda x: self._esd_first_term_integrand_func(np.array([x]), rflats[i:i+1]),
                MIN_INTEGRATION_RADIUS,
                rflats[i],
            ) for i in range(self.radii.size)],
            dtype=object,
        )
        _check_errors_ok(first_term_integral[:, 1].astype(float))
        return np.concatenate(first_term_integral[:, 0]).astype(float)

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
