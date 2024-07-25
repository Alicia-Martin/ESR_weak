import numpy as np
from scipy.optimize import minimize, basinhopping

# Define the objective function
def rosenbrock(x):
    return sum(100.0 * (x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)

# Initial guess
x0 = np.array([1.3, 0.7])

# Set up the local optimizer
minimizer_kwargs = {"method": "L-BFGS-B"}

# Perform basin hopping
result = basinhopping(rosenbrock, x0, minimizer_kwargs=minimizer_kwargs, niter=200)

# Print the result
print("Global minimum: x = ", result.x, ", f(x) = ", result.fun)
