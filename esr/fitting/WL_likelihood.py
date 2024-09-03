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



class WLLikelihood(Likelihood):
    def __init__(self, data_file, run_name, data_dir=None, fn_set = 'core_maths'):
        """Likelihood class used to fit a function directly using a Gaussian likelihood
 
        # """
        # print('run_name', run_name)
        #Load data
        # Read the file
        # with open(data_file, 'r') as file:
        #     lines = file.readlines()

        with open(data_file, 'rb') as file:
            data = pickle.load(file)

        no_nans = np.isnan(data['ds']) == False
        self.xvar = data['rp'][no_nans]
        self.yvar = data['ds'][no_nans]
        self.yerr = data['ds_err'][no_nans]

        # print(self.xvar, self.yvar, self.yerr)
        # sys.exit()
        # Extract the data
        # self.xvar = jnp.array([float(value) for value in lines[0].strip().split(', ')])
        # self.yvar = jnp.array([float(value) for value in lines[1].strip().split(', ')])
        # self.yerr = jnp.array([float(value) for value in lines[2].strip().split(', ')])


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
    
    def get_pred(self, a, xvar, eq_numpy):
        # if len(a) == 0:
        #         rho_fun = eq_numpy(xvar)
        # else:
        #         rho_fun = eq_numpy(xvar, *a)

        eds = ExcessSurfaceDensity.calculate(xvar, eq_numpy, params=a)
        return eds
    

    def get_loss(self, eq_numpy, verbose=False, value='value_and_grad', **kwargs):
        # @profile

        def check_density(params, eq_numpy, r):
            if len(params) == 0:
                wrapped_eq_diff = eq_numpy(r)
            else:
                wrapped_eq_diff = eq_numpy(r, *params)
            return jnp.where(jnp.all(wrapped_eq_diff >= 0), 0, np.inf)
        
        def f_loss(a, xvar, yvar, yerr):
            ypred = self.get_pred(a, xvar, eq_numpy)

            #check that the density is positive
            negloglike = 0
            # negloglike += check_density(a, eq_numpy, xvar)


            def neg_log_gaussian(x, mean, std):
                return (x - mean)**2/(2*std**2)

            nll = neg_log_gaussian(ypred, yvar, yerr)
            nll = jnp.sum(nll)

            negloglike += nll

            # jax.debug.print('{x}', x = negloglike)
            
            return negloglike
        
        if value == 'hessian':
            return jax.hessian(f_loss)
        
        elif value == 'evaluate':
            return f_loss
        
        elif value == 'grad':
            return jax.grad(f_loss)

        else:
            return jax.jit(jax.value_and_grad(f_loss))
    
    def get_wrapped_like(self, loss_template):
        
        def handle_nans(negloglike):
            try:
                return jnp.where(jnp.isnan(negloglike), np.inf, negloglike)
            except:
                return (jnp.where(jnp.isnan(negloglike[0]), np.inf, negloglike[0]),) + negloglike[1:]

        def wrapped_like(x, xvar, yvar, yerr, signs=None, check_nans =True):
            if signs is None:
                p = x.copy()
            else:
                p = x.copy()
                try:
                    p[:len(signs)] = [s * 10**xi for s, xi in zip(signs, x)]
                except:
                    p = p.at[:len(signs)].set([s * 10**xi for s, xi in zip(signs, x[:len(signs)])])
            
            loss = loss_template(p, xvar, yvar, yerr)

            if check_nans:
                loss = handle_nans(loss)

            return loss
        return wrapped_like

