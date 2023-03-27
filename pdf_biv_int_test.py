# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 12:08:33 2023

@author: Anderson Borba
"""
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
#
import polsar_pdfs as pp
#import polsar_plot as pplt
#import polsar_total_loglikelihood as ptl
#
plt.rc('text', usetex=True)
fig = plt.figure()
#ax = fig.gca(projection='3d')
ax = Axes3D(fig)
rho = 0.1
L = 2
s1 = 0.05
s2 = 0.02
m = 10
z1 = np.zeros(m)
z2 = np.zeros(m)
a = 0 
b = 1
h = (b - a) / m
for i in range(m):
    z1[i] = a + h * i
    z2[i] = a + h * i
s = (m, m)
fz = np.zeros(s) 
for i in range(m): 
    for j in range(m):
        #fz[i, j] = pp.pdf_ratio_intensities(z1[i], z2[j], rho, L, s1, s2) 
        fz[i, j] = 1
print(fz)
#            
x1, y1 = np.meshgrid(z1, z2)
surf = ax.plot_surface(x1, y1, fz, cmap=cm.coolwarm, linewidth=0, antialiased=False)
#plt.xlabel(r'$\sigma_1$')
#plt.ylabel(r'$\sigma_2$')
#plt.title(r'PDF Bivariate Product of Intensities$')
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()


