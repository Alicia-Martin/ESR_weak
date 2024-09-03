
import numpy as np
from esr.esd import ExcessSurfaceDensity
import sympy
from esr.fitting.sympy_symbols import *
from scipy.optimize import minimize
import sys
import matplotlib.pyplot as plt

#Open data
data_file = 'esr/dark_matter_data.txt'
with open(data_file, 'r') as file:
    lines = file.readlines()

# Extract the data
xvar = np.array([float(value) for value in lines[0].strip().split(', ')])
yvar = np.array([float(value) for value in lines[1].strip().split(', ')])
yerr = np.array([float(value) for value in lines[2].strip().split(', ')])


#Fun to optimise
nparam = 2
fn = 'a0/(x*(x + a1))^2'
fcn_i = fn.replace('\n', '')
fcn_i = fn.replace('\'', '')


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
print(eq)
eq_numpy = sympy.lambdify([x, a0], eq, modules=["numpy"])


if nparam > 1:
    all_a = ' '.join([f'a{i}' for i in range(nparam)])
    all_a = list(sympy.symbols(all_a, real=True))
    eq_numpy = sympy.lambdify([x] + all_a, eq, modules=["numpy"])
else:
    eq_numpy = sympy.lambdify([x, a0], eq, modules=["numpy"])
# print(eq_numpy)
# print(eq_numpy(1, 1))
# sys.exit()

#Optimise
def objective_function(params, radii, yvar, yerr, density_func):
    # params = 10**params
    esd = ExcessSurfaceDensity.calculate(radii, density_func, params=params)
    
    return np.sum((esd - yvar)**2/(2*yerr**2))

initial_params = [1.0, 1.0]  # Initial guess for the parameters

result = minimize(objective_function, initial_params, args=(xvar, yvar, yerr, eq_numpy), method='Nelder-Mead')
final_eds = ExcessSurfaceDensity.calculate(xvar, eq_numpy, params=result.x)

print(result.x)

plt.errorbar(xvar, yvar, yerr=yerr, fmt='o')
plt.plot(xvar, final_eds, label='Fit')
plt.yscale("log")
plt.xscale("log")
plt.show()

