import numpy as np
import os
import sys
from mpi4py import MPI
import pickle
import scipy
from sympy import *
import sympy
import time
import matplotlib.pyplot as plt
import jax.numpy as jnp

import esr.fitting.test_all
import esr.fitting.test_all_Fisher
import esr.fitting.match
import esr.fitting.combine_DL
import esr.fitting.plot
from esr.fitting.likelihood import Likelihood
from esr.generation.simplifier import time_limit
from esr.fitting.sympy_symbols import *
import esr.plotting.plot
import esr.generation.simplifier as simplifier
from esr.fitting.fit_single import fit_from_string
from esr.fitting.WL_likelihood import WLLikelihood

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def fit_galaxy(data_file, run_name, comp, try_integration=False, method="Nelder-Mead", log_opt=True):
    if log_opt:
        log = '_log'
    else:
        log = ''

    #Likelihood
    likelihood = WLLikelihood(data_file, run_name + log, data_dir=None, fn_set = 'core_maths')   
    
    #run esr
    esr.fitting.test_all.main(comp, likelihood, try_integration=try_integration, log_opt=log_opt, method=method)
    esr.fitting.test_all_Fisher.main(comp, likelihood, tmax=5, try_integration=try_integration)
    esr.fitting.match.main(comp, likelihood, tmax=5, try_integration=try_integration)
    esr.fitting.combine_DL.main(comp, likelihood)
    # esr.fitting.plot.main(comp, likelihood, tmax=5, try_integration=try_integration)
    
###########################################################################
                                 # MAIN #
###########################################################################
    
#------------------------------------------------------------
# run code for a single set of data
#------------------------------------------------------------

comp = 5
try_integration = False
method = "BFGS"
log_opt = False
data_file = 'dark_matter_data.txt'
run_name = 'WL'

if rank == 0:
    print('method:', method, ', log_opt:', log_opt, flush=True)

print(data_file)
sys.exit(0)

start = time.time()
fit_galaxy(data_file, run_name, comp, try_integration=try_integration, method=method, log_opt=log_opt)
end = time.time()

if rank == 0:
    print('Total time taken:', end - start, flush=True)