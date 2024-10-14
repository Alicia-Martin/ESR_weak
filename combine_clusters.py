# Combine data from all clusters
import numpy as np
import matplotlib.pyplot as plt

galaxy_names_file = '10_clusters.txt'

#open all files in directory XXL with these names
with open(galaxy_names_file) as f:
    names = f.read().splitlines()


x = []
y = []
yerr = []
for name in names:
    print(name)
    datafile= 'XXL/' + name + '.txt'
    with open(datafile, 'r') as file:
            lines = file.readlines()

    # Extract the data
    yvar = np.array(list(map(float, lines[1].split())))
    xvar = np.array(list(map(float, lines[0].split())))
    yerr_value = np.array(list(map(float, lines[2].split())))

    x = np.append(x, xvar)
    y = np.append(y, yvar)
    yerr = np.append(yerr, yerr_value)


#Take out nans
no_nans = np.isnan(y) == False
x = x[no_nans]
y = y[no_nans]
yerr = yerr[no_nans]

#order the data in terms of incerasing x
sorted_indices = np.argsort(x)
x = x[sorted_indices]
y = y[sorted_indices]
yerr = yerr[sorted_indices]


#save the data in the sme format as the original files
datafile= 'XXL/combined_data.txt'
with open(datafile, 'w') as file:
    file.write(' '.join(map(str, x)) + '\n')
    file.write(' '.join(map(str, y)) + '\n')
    file.write(' '.join(map(str, yerr)) + '\n')


#open saved file and plot it
datafile= 'XXL/combined_data.txt'
with open(datafile, 'r') as file:
        lines = file.readlines()

# Extract the data
yvar = np.array(list(map(float, lines[1].split())))
xvar = np.array(list(map(float, lines[0].split())))
yerr_value = np.array(list(map(float, lines[2].split())))

plt.errorbar(xvar, yvar, yerr=yerr_value, fmt='o')
plt.show()


