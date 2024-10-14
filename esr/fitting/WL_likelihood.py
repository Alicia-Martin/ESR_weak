from esr.fitting.likelihood import Likelihood
import jax.numpy as jnp
import sympy
from sympy import symbols
import jax
import numpy as np
from esr.fitting.sympy_symbols import *

from esr.esd import ExcessSurfaceDensity
import sys
import pickle
import matplotlib.pyplot as plt

import numpy as np
from scipy.integrate import quad
from scipy.optimize import fsolve



class WLLikelihood(Likelihood):
    def __init__(self, data_file, run_name, data_dir=None, fn_set = 'core_maths'):
        """Likelihood class used to fit a function directly using a Gaussian likelihood
 
        # """

        # with open(data_file, 'rb') as file:
        #     data = pickle.load(file)


        # no_nans = np.isnan(data['ds']) == False
        # self.xvar = data['rp'][no_nans]
        # self.yvar = data['ds'][no_nans]
        # self.yerr = data['ds_err'][no_nans]
        # print('run_name', run_name)

        #Load data
        # Read the file
        with open(data_file, 'r') as file:
            lines = file.readlines()

        # Extract the data and take out nans
        
        self.yvar = jnp.array(list(map(float, lines[1].split())))
        # no_nans = np.isnan(self.yvar) == False
        # self.yvar = self.yvar[no_nans]
        self.xvar = jnp.array(list(map(float, lines[0].split())))
        self.yerr = jnp.array(list(map(float, lines[2].split())))

        super().__init__(data_file, data_file, run_name, data_dir=data_dir, fn_set = fn_set)
        self.ylabel = r'$y$'    # for plotting

    def run_sympify(self, fcn_i, tmax=5, try_integration=True):
        fcn_i = fcn_i.replace('\n', '')
        fcn_i = fcn_i.replace('\'', '')


        eq = sympy.sympify(fcn_i,
                    locals={"inv": inv,
                           "square": square,
                            "cube": cube,
                            "sqrt": sqrt,
                            "log": log,
                            "pow": pow,
                            "x": x,
                            "a0": a0,
                            "a1": a1,
                            "a2": a2})
        
        return fcn_i, eq
    
    def M_delta(self, delta, fcn_i, params):
        # need to add the redshift as an argument

        # Constants
        G = 6.67430e-11  # Gravitational constant in m^3 kg^-1 s^-2
        H0 = 70.0 * 1e3 / (3.086e22)  # Hubble constant in s^-1 (H0 = 70 km/s/Mpc)

        # Define critical density at redshift z
        def rho_crit(z=0):
            H_z = H0 * np.sqrt(0.3 * (1 + z)**3 + 0.7)  # Flat Lambda-CDM model
            rho_crit_si = (3 * H_z**2) / (8 * np.pi * G)  # Critical density in kg/m³
            conversion_factor = 1.477e37  # Convert kg/m³ to M_sun/Mpc³
            return rho_crit_si * conversion_factor  # Critical density in M_sun/Mpc³

        # Define the density profile
        def rho(fcn_i, params, r):
            rho = fcn_i(r, *params)*1e12 #units of solar masses per cubic Mpc
            return rho

        # Mass enclosed within radius r
        def mass_enclosed(r):
            mass_integral, _ = quad(lambda r_prime: 4 * np.pi * rho(fcn_i, params, r_prime) * r_prime**2, 0, r)
            return mass_integral  # In solar masses

        # Equation for finding r200
        def find_r_delta(r, delta):
            M_r = mass_enclosed(r)  # Mass within radius r
            rho_mean = M_r / ((4 / 3) * np.pi * r**3)  # Mean density within radius r
            return rho_mean - delta * rho_crit()  # Difference from 200 * rho_crit

        # Solve for r200
        r_delta_guess = 1.0  # Initial guess in Mpc
        r_delta = fsolve(find_r_delta, r_delta_guess, args=(delta))[0]

        # Calculate M200
        M_delta = mass_enclosed(r_delta)

        print(f"r200 = {r_delta:.3f} Mpc")
        print(f"M200 = {M_delta:.3e} solar masses")

        return M_delta, r_delta

    
    def get_pred(self, a, xvar, eq_numpy):
        # if len(a) == 0:
        #         rho_fun = eq_numpy(xvar)
        # else:
        #         rho_fun = eq_numpy(xvar, *a)
        # a = jnp.array([1])
        # xvar = [1,1]
        # print(eq_numpy(xvar, *a))
        # sys.exit()

        eds = ExcessSurfaceDensity.calculate(xvar, eq_numpy, params=a)
        # jax.debug.print('{x}, {y}', x = eds, y=a)
        return eds
    

    def get_loss(self, eq_numpy, verbose=False, value='value_and_grad', **kwargs):
        # @profile

        # def check_density(params, eq_numpy, r):
        #     #this is wrong, it's cheacking the density at the projected radius not the 3d radius
        #     xs = jnp.linspace(0, r[-1], 200)
        #     # sys.exit()
        #     if len(params) == 0:
        #         wrapped_eq_diff = eq_numpy(xs)
        #     else:
        #         wrapped_eq_diff = eq_numpy(xs, *params)
        #     return jnp.where(jnp.all(wrapped_eq_diff >= 0), 0, np.inf)
        
        def f_loss(a, xvar, yvar, yerr):
            # a = [5.0, -0.2]
            
            ypred = self.get_pred(a, xvar, eq_numpy)
            #check that the density is positive
            negloglike = 0

            def neg_log_gaussian(x, mean, std):
                return (x - mean)**2/(2*std**2)

            nll = neg_log_gaussian(ypred, yvar, yerr)
            nll = jnp.sum(nll)

            negloglike += nll

            # jax.debug.print('{x}, {y}', x = negloglike, y=a)
            # sys.exit()
            return negloglike
        
        if value == 'hessian':
            return jax.hessian(f_loss)
        
        elif value == 'evaluate':
            return f_loss
        
        elif value == 'grad':
            return jax.grad(f_loss)

        else:
            return jax.jit(jax.value_and_grad(f_loss))
    
    def get_wrapped_like(self, loss_template, signs=None):
        
        def handle_nans(negloglike):
            try:
                return np.where(np.isnan(negloglike), np.inf, negloglike)
            except:
                return (np.where(np.isnan(negloglike[0]), np.inf, negloglike[0]),) + negloglike[1:]

        def wrapped_like(x, xvar, yvar, yerr, signs=None, check_nans =True):
            if signs is None:
                p = x.copy()
            else:
                p = x.copy()
                try:
                    p[:len(signs)] = [s * 10**xi for s, xi in zip(signs, p)]
                except:
                    p = p.at[:len(signs)].set([s * 10**xi for s, xi in zip(signs, p[:len(signs)])])

                # self.yvar = jnp.array(list(map(float, lines[1].split())))

            loss = loss_template(p, xvar, yvar, yerr)
            # loss = np.nan
            # print(loss)

            # if check_nans:
            #     loss = handle_nans(loss)
            # print(loss)

            return loss
        return wrapped_like
    

    def get_wrapped_like(self, loss_template, signs=None):
        def wrapped_like(x, xvar, yvar, yerr, signs=None, global_params=None, global_index=None):
            #mixed global and local params
            global_local = np.zeros(len(global_params) + len(x))
            global_local[global_index] = global_params
            global_local[global_index:] = x
            if signs is None:
                p = global_local.copy()
            else:
                p = global_local.copy()
                try:
                    p[:len(signs)] = [s * 10**xi for s, xi in zip(signs, p)]
                except:
                    p = p.at[:len(signs)].set([s * 10**xi for s, xi in zip(signs, p[:len(signs)])])

            loss = loss_template(p, xvar, yvar, yerr)
            return loss
        return wrapped_like

