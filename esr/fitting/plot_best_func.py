import numpy as np
import matplotlib.pyplot as plt
from esr.esd import ExcessSurfaceDensity
from esr.fitting.sympy_symbols import *
import esr.generation.simplifier as simplifier
import matplotlib.cm as cm
from esr.fitting.WL_likelihood import WLLikelihood
from matplotlib.backends.backend_pdf import PdfPages



def plot_func(ax, likelihood, fcn, params, L_value, name):
    fcn_i = fcn.replace('\'', '')
    max_param = 4
    
    k = simplifier.count_params([fcn_i], max_param)[0]
    measured = params[:k]

    if not np.any(params[k:] == 0):
        print('NICE', name)
    
    # try:
    fcn_i, eq= likelihood.run_sympify(fcn_i)
    
    if k == 0:
        eq_numpy = sympy.lambdify([x], eq, modules=["numpy"])
    elif k > 1:
        all_a = ' '.join([f'a{i}' for i in range(k)])
        all_a = list(sympy.symbols(all_a, real=True))
        eq_numpy = sympy.lambdify([x] + all_a, eq, modules=["numpy"])
    else:
        eq_numpy = sympy.lambdify([x, a0], eq, modules=["numpy"])

    #Plot the ESD on the first figure        
    # x_array = np.linspace(0, likelihood.xvar.max(), 1000)
    x_array = np.linspace(0, likelihood.xvar.max()+ 3, 1000)

    esd = ExcessSurfaceDensity.calculate(x_array, eq_numpy, params=measured)
    ypred = eq_numpy(x_array, *measured)

    # if NFW and i == 0:
    #     ax1.plot(x_array, esd, color='blue', zorder=len(fcn_list)-i, label='NFW')
    # else:
    if np.isscalar(esd):
        ax.plot(x_array, [esd]*len(likelihood.xvar))
    else:
        ax.plot(x_array, esd)
        ax.plot(x_array, ypred, label=fcn_i)

    ax.set_xlabel('R [Mpc]')
    ax.set_ylabel('Excess Surface Density')
    ax.legend()

    #Add title
    ax.set_title(name, L_value)



def main():
    comp = 7
    dirname = 'output_decreasing/output/'
    max_params = 4
    with open('all_clusters.txt') as f:
            names = f.read().splitlines() 

    pdf_filename = "galaxy_plots_.pdf"
    with PdfPages(pdf_filename) as pdf:
        for name in names:
                data_file = 'XXL/' + str(name) + '.txt'
                run_name = 'WL_' + str(name)
                likelihood = WLLikelihood(data_file, run_name)


                try:
                    data = np.genfromtxt(dirname + 'output_WL_' + name +  '/combine_DL_comp' + str(comp) + '.dat')  
                except:
                    data = np.genfromtxt(dirname + 'output_WL_' + name +  '_all_fun/combine_DL_comp' + str(comp) + '.dat') 

                # Read the equations file
                with open(dirname + 'output_WL_' + name +  '/combine_DL_fcn_comp' + str(comp) + '.dat', 'r') as file:
                    fcn_variant = file.readlines()

                #Select the function that I want to plot
                idx = 1419
                p_actual = data[idx, 1:1+ max_params]
                delta_actual = data[idx, 1+max_params: -6]
                L_value = data[idx,0]
                # print(fcn_variant[idx], p_actual)

                fig, ax = plt.subplots(figsize=(8, 6))
                plot_func(ax, likelihood, fcn_variant[idx], p_actual, L_value, name)

                # Save the current plot to the PDF
                pdf.savefig(fig)
                plt.close(fig)

    print(f"Plots saved in {pdf_filename}")


if __name__ == "__main__":
    main()
         