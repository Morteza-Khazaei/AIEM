import os
import random
import itertools
from aiem import AIEM
import numpy as np
import matplotlib.pyplot as plt



fname = r'LC08_032032_20220912_eps_5.3GHz'
base_path = r'D:\PhD source code\datasets'
input_path = os.path.join(base_path, 'outputs')
# output_path = os.path.join(base_path, 'outputs')
infile = os.path.join(input_path, fname + '.npz')
# outfile = os.path.join(output_path, oname + '.npz')

# Load Array
dictData = np.load(infile)
stack = dictData['arr_0']

# Print the array
nsize, xsize, ysize = stack.shape
print(nsize, xsize, ysize)

frq = 5.3
itype = ['2',]
theta_i = np.arange(5, 85, 5)
theta_s = np.arange(5, 85, 5)
phi_i = np.arange(5, 185, 5)
phi_s = np.arange(5, 185, 5)
sigma = np.arange(0.001, 0.05, 0.005)
print(sigma)
cl = np.arange(0.001, 0.1, 0.005)
print(cl)

# to compute all possible permutations
permutation = list(itertools.product(*[theta_i, theta_s, phi_i, phi_s, sigma, cl]))
print(len(permutation))

bnames = ['HH', 'VH', 'HV', 'VV']
id_eps = random.randint(0, nsize)
print(id_eps)
id_per = random.randint(0, len(permutation))
print(id_per)

er = stack[id_eps]
p = permutation[id_per]
sig = AIEM(frq_ghz=frq, theta_i=p[0], theta_s=p[1], phi_i=p[2], phi_s=p[3], sigma=p[4], cl=p[5], eps=er, itype='2').run()

# Plot the SOM band
fig, axs = plt.subplots(2, 2, figsize=(8, 8))
for i, ax in enumerate(axs.ravel()):
    im = ax.imshow(sig[i], cmap='Spectral_r')
    ax.set_title(rf'$\sigma_{{{bnames[i]}}}$', fontsize=10)
    fig.colorbar(im, ax=ax)

fig.suptitle(rf'$\theta_i={p[0]}; \theta_s={p[1]}; \phi_i={p[2]}; \phi_s={p[3]}$')
fig.tight_layout()
plt.show()