import os
import re
import numpy as np
import matplotlib.pyplot as plt
import csv

def pareto_plot(dirname, savename, prefix='final', do_DL=True, do_logL=True):
    """
    Plot the pareto front using the files in a given directory
    
    Args:
        :dirname (str): The directory name to consider.
        :savename (str): File name to save file (within dirname).
        :prefix (str, default='final'): Prefix of the files to process ('final' or 'final_katz').
        :do_DL (bool, default=True): Whether to plot the description length in the pareto front.
        :do_logL (bool, default=True): Whether to plot the log-likelihood in the pareto front.
    """
    
    if (not do_DL) and (not do_logL):
        return

    all_f = os.listdir(dirname)
    all_f = [f for f in all_f if f.startswith(prefix)]
    all_f.sort(key=lambda f: int(re.search(r'\d+', f).group()))  # Sort files numerically

    all_comp = [int(f[len(prefix) + 1:-len('.dat')]) for f in all_f]  # Extract complexity
    all_logL = np.empty(len(all_comp))
    all_DL = np.empty(len(all_comp))
    
    for i, fname in enumerate(all_f):
        print(i, fname)
        
        with open(os.path.join(dirname, fname), 'r') as f:
            reader = csv.reader(f, delimiter=';')
            data = [row for row in reader]
            data = np.array([d[2:7] for d in data], dtype=float)
            
        # Get min DL and logL
        try:
            min_DL_index = np.nanargmin(data[:, 0])  # Index of minimum DL
            min_logL_index = np.nanargmin(data[:, 2])  # Index of minimum logL
            all_DL[i] = np.nanmin(data[:, 0])
            all_logL[i] = np.nanmin(data[:, 2])
        except:
            all_DL[i] = np.nan
            all_logL[i] = np.nan
    
    # Normalize DL and logL
    all_DL -= np.nanmin(all_DL)
    all_logL -= np.nanmin(all_logL)
    
    # Filter out NaN values
    m = np.isfinite(all_DL)
    all_comp = np.array(all_comp, dtype=int)[m]
    all_DL = all_DL[m]
    all_logL = all_logL[m]

    return all_comp, all_DL, all_logL

# Function to plot both final and final_katz on the same plot
def plot_final_and_katz(dirname, savename):
    # Get data for 'final' files
    comp_final, DL_final, logL_final = pareto_plot(dirname, savename, prefix='final')
    # Get data for 'final_katz' files
    comp_katz, DL_katz, logL_katz = pareto_plot(dirname, savename, prefix='final_katz')

    # Plotting
    fig, ax1 = plt.subplots(1, 1, figsize=(5, 3.5))
    cm = plt.get_cmap('Set1')

    # Plot DL for final and final_katz
    ax1.plot(comp_final, DL_final, marker='.', color=cm(0), markersize=5, label='final DL')
    ax1.plot(comp_katz, DL_katz, marker='.', color=cm(2), markersize=5, label='final_katz DL')
    ax1.set_ylabel(r'$\Delta L \left( D \right)$')
    ax1.set_xlabel(r'Complexity')
    ax1.yaxis.label.set_color(cm(0))
    ax1.tick_params(axis='y', colors=cm(0))
    ax1.spines['left'].set_color(cm(0))

    # Plot logL for final and final_katz on a secondary y-axis
    ax2 = ax1.twinx()
    ax2.plot(comp_final, logL_final, marker='.', color=cm(1), markersize=5, label='final logL')
    ax2.plot(comp_katz, logL_katz, marker='.', color=cm(3), markersize=5, label='final_katz logL')
    ax2.set_ylabel(r'$ \left| \Delta \log\mathcal{L} \right|$')
    ax2.yaxis.label.set_color(cm(1))
    ax2.tick_params(axis='y', colors=cm(1))
    ax2.spines['right'].set_color(cm(1))

    # Add legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    # Save and show the plot
    fig.tight_layout()
    fig.savefig(os.path.join(dirname, savename), bbox_inches='tight', dpi=300)
    plt.show()

# Example usage
dirname = 'esr/fitting/output_decreasing/output/combining_clusters/combining_clusters/'
savename = 'esr/fitting/output_decreasing/output/combining_clusters/combining_clusters/combined_pareto_plot.png'
plot_final_and_katz(dirname, savename)