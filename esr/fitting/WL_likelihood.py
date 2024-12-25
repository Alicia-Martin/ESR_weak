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

import jax.numpy as jnp

from quadax import quadgk



class WLLikelihood(Likelihood):
    def __init__(self, data_file, run_name, data_dir=None, fn_set = 'core_maths', keep_nans = False, physicalize = False):
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
        if keep_nans:
            self.yvar = jnp.array(list(map(float, lines[1].split())))
            self.xvar = jnp.array(list(map(float, lines[0].split())))
            self.yerr = jnp.array(list(map(float, lines[2].split())))
        else:
            self.yvar = jnp.array(list(map(float, lines[1].split())))
            no_nans = np.isnan(self.yvar) == False
            self.yvar = self.yvar[no_nans]
            self.xvar = jnp.array(list(map(float, lines[0].split())))[no_nans]
            self.yerr = jnp.array(list(map(float, lines[2].split())))[no_nans]

        # print(data_file)
        # print('xvar', len(self.xvar)/9)
        # sys.exit()
        # print('yvar', self.yvar)
        # print('yerr', self.yerr)
            
        self.physicalize = physicalize

        super().__init__(data_file, data_file, run_name, data_dir=data_dir, fn_set = fn_set)
        self.ylabel = r'$y$'    # for plotting

    def physicalize_fn(self, eq, rho0, rs):
        #add a scale radius and characterisitic density to the density fucntions
        eq_transformed = rho0 * eq.subs(x, x / rs)
        return eq_transformed


    def run_sympify(self, fcn_i, tmax=5, try_integration=True):
        fcn_i = fcn_i.replace('\n', '')
        fcn_i = fcn_i.replace('\'', '')

        rs = sympy.Symbol('rs')  # Scale radius
        rho0 = sympy.Symbol('rho0')  # Normalization constant


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
                            "a2": a2,
                            "rs": rs,
                            "rho0": rho0,
                            })
        
        if self.physicalize:
            eq = self.physicalize_fn(eq, rho0, rs)
     
        self.fcn_i = fcn_i
        self.eq = eq
        
        return fcn_i, eq
    
    def M_delta(self, delta, fcn_i, params, uncertainties):
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
        def mass_enclosed(r, params):
            fun = lambda r_prime: 4 * jnp.pi * rho(fcn_i, params, r_prime) * r_prime**2
            interval = jnp.array([0, r])
            mass_integral = quadgk(fun,  interval)

            #mass integral with quad
            # mass_integral = quad(fun, 0, r)

            mass_integral = mass_integral[0]
            return mass_integral  # In solar masses

        # Equation for finding r200
        def find_r_delta(r, delta, params):
            r = r[0]
            M_r = mass_enclosed(r, params)  # Mass within radius r
            rho_mean = M_r / ((4 / 3) * np.pi * r**3)  # Mean density within radius r
            return rho_mean - delta * rho_crit()  # Difference from 200 * rho_crit

        # Solve for r200
        r_delta_guess = 1.0  # Initial guess in Mpc
        r_delta = fsolve(find_r_delta, r_delta_guess, args=(delta, params))[0]

        # Calculate M200
        M_delta = mass_enclosed(r_delta, params)

        def M_delta_at_r_delta(params):
            return mass_enclosed(jnp.array(r_delta), params)


        #Calculate the uncertainty
        def calculate_Delta(params):
            # Calculate the Hessian
            hessian_template = self.get_loss(fcn_i, value = 'hessian')
            Hmat = hessian_template(params, self.xvar, self.yvar, self.yerr)
            # print('Hmat', Hmat)
            
            #Other related quantities
            Fisher_diag = jnp.diag(Hmat)
            Delta = np.sqrt(12./Fisher_diag)
            return Delta
        
        # params_uncertainties = calculate_Delta(params)

        params_uncertainties = uncertainties

        # Calculate uncertainties using the gradient
        M_uncertainty_squared = 0
        gradient_values = jax.grad(M_delta_at_r_delta)(jnp.array(params))

        for i in range(len(params)):
            M_uncertainty_squared += (gradient_values[i] * params_uncertainties[i]) ** 2

        M_uncertainty = np.sqrt(M_uncertainty_squared)
        
        print(f"r200 = {r_delta:.3f} Mpc")
        print(f"M200 = {M_delta:.3e} solar masses")
        print(f"M200 uncertainty = {M_uncertainty:.3e} solar masses")

        return M_delta, r_delta, M_uncertainty

    def get_pred(self, a, xvar, eq_numpy):
        # if len(a) == 0:
        #         rho_fun = eq_numpy(xvar)
        # else:
        #         rho_fun = eq_numpy(xvar, *a)
        # a = jnp.array([1])
        # xvar = [1,1]
        # print(eq_numpy(xvar, *a))
        # jax.debug.print({}, eq_numpy(jnp.array([10.]), *a))
        # sys.exit()

        # print(eq_numpy(0.5682318, 10.65178197, 18.75110151))
        # a = [-2, 1]
        eds = ExcessSurfaceDensity.calculate(xvar, eq_numpy, params=a)
        # print(eds)
        # sys.exit()
        # jax.debug.print('{x}, {y}', x = eds, y=a)
        return eds
    

    def get_loss(self, eq_numpy, verbose=False, value='value_and_grad', **kwargs):
        def check_decreasing_function(params):           
            grad_density = jax.grad(eq_numpy, argnums=0)(10., *params)
            # jax.debug.print('grad {x}', x = grad_density)
            return jnp.less(grad_density, 0)
        
        def combined_params(params, global_params, global_index):
            # Initialize the global local array
            combined_params = jnp.zeros(len(global_params) + len(params), dtype=global_params.dtype)
            
            # put the global params in the right place
            global_index_array = jnp.array(global_index)
            combined_params = combined_params.at[global_index_array].set(global_params)

            # put the local params in the right place
            all_indices = jnp.arange(combined_params.shape[0])
            is_local = jnp.isin(all_indices, global_index_array, invert=True)
            local_indices = jnp.nonzero(is_local, size=len(params))[0]  
            combined_params = combined_params.at[local_indices].set(params)

            return combined_params
                
        def f_loss(a, xvar, yvar, yerr, global_params=None, global_index=None):

            if global_params is not None:
                a = combined_params(a, global_params, global_index)

            negloglike = 0

            #check decreasing density
            # decrease = check_decreasing_function(a)
            # negloglike = jax.lax.cond(decrease, lambda: 0.0, lambda: jnp.inf)
            # jax.debug.print('decrease {x} {decrease}', x = a, decrease=decrease)

            ypred = self.get_pred(a, xvar, eq_numpy)
            # jax.debug.print('ypred {ypred}, yvar {yvar}, yerr {yerr}', ypred=ypred, yvar=yvar, yerr=yerr)
            #check that the density is positive


            def neg_log_gaussian(x, mean, std):
                return (x - mean)**2/(2*std**2)

            nll = neg_log_gaussian(ypred, yvar, yerr)
            nll = jnp.sum(nll)

            negloglike += nll
            # jax.debug.print('negloglike {negloglike}', negloglike=negloglike)
            # sys.exit()
            return negloglike
        # print(eq_numpy(1,1,1))

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
            

        def check_divergences(params):
            a_symbols = [a0, a1, a2, a3]
            substitutions = {a_symbols[i]: params[i] for i in range(len(params))}
            eq_substituted = self.eq.subs(substitutions)
            print(params)

            singularities = sympy.singularities(eq_substituted, x, domain=sympy.Interval(0, sympy.oo))

            # continuous = sympy.calculus.util.continuous_domain(eq_substituted, x, domain=sympy.Interval(0, sympy.oo))
            # print(continuous)

            if isinstance(singularities, sympy.ConditionSet):
                has_solutions = True
            else:
                has_solutions = len(singularities) > 0

            is_finite = not has_solutions

            # is_finite
            print(is_finite)
            # is_finite = eq_substituted.is_finite
            return is_finite

        def wrapped_like(x, xvar, yvar, yerr, signs=None, global_params=None, global_index=None, check_nans=False):
            if signs is None:
                p = x.copy()
            else:
                p = x.copy()
                try:
                    p[:len(signs)] = [s * 10**xi for s, xi in zip(signs, p)]
                except:
                    p = p.at[:len(signs)].set([s * 10**xi for s, xi in zip(signs, p[:len(signs)])])

            # if check_divergences(p):
            #     # print('here')
            #     loss = loss_template(p, xvar, yvar, yerr, global_params, global_index)
            #     # print('loss', loss[0])
            # else:
            #     loss = (jnp.inf, jnp.zeros(len(p)))

            # p = [1., 0.]
            loss = loss_template(p, xvar, yvar, yerr, global_params, global_index)

            # print('loss', loss)
    
            # sys.exit()

            # loss = np.nan
            # print('loss', loss[0])

            # if check_nans:
            #     loss = handle_nans(loss)
            # print(loss)

            return loss
        return wrapped_like


