## Version 06/01/2023
# Article GRSL + Remote sensing 2023
# Ref:
# A. A. De Borba,
# M. Marengoni and
# A. C. Frery,
# "Fusion of Evidences in Intensity Channels for Edge Detection in PolSAR Images,"
# in IEEE Geoscience and Remote Sensing Letters,
#doi: 10.1109/LGRS.2020.3022511.
# bibtex
#@ARTICLE{9203845,
#  author={De Borba, Anderson A. and Marengoni, Maurício and Frery, Alejandro C.},
#  journal={IEEE Geoscience and Remote Sensing Letters},
#  title={Fusion of Evidences in Intensity Channels for Edge Detection in PolSAR Images},
#  year={2020},
#  volume={},
#  number={},
#  pages={1-5},
#  doi={10.1109/LGRS.2020.3022511}}
#
import numpy as np
## Used to present the images
#import matplotlib as mpl
#import matplotlib.pyplot as plt
## Used to find border evidences
import math
from scipy.special import iv, kv
#from scipy.optimize import dual_annealing
#### Used to find_evidence_bfgs
#from scipy.optimize import minimize
#
#import polsar_pds as pp
#import polsar_loglikelihood as plk
#import polsar_total_loglikelihood as ptl
#import polsar_plot as pplt
#
def pdf_ratio_intensities(z1, z2, rho, L, s1, s2):
    div1 = z1 / s1
    div2 = z2 / s2
    arg1 = - L * (div1 + div2 ) / (1 - rho**2) 
    aux1 = L**(L+1) * (z1 * z2)**(0.5 * (L - 1)) * np.exp(arg1)
    aux2 = (s1 * s2)**(0.5 * (L + 1)) * math.gamma(L) * (1 - rho**2) * rho**(L - 1)
    arg2 = 2 * L * np.sqrt((z1 * z2) / (s1 * s2)) * (rho / (1 - rho**2))
    #pdf = aux1 * aux2 * iv(L - 1, arg2)
    #print(aux2)
    pdf = aux1 * iv(L - 1, arg2) / aux2 
    #pdf = np.abs(np.cos(z1) + np.cos(z2))
    return pdf
#
def pdf_prod_intensities(z, L, rho, mu1, mu2):
    h = mu1 * mu2
    div1 = 4 * L**(L + 1) * z**L
    div2 = math.gamma(L) * (1 - rho**2) * h**(L + 1)
    aux = div1 / div2
    arg1 = 2 * rho * L * z / ((1 - rho**2) * h)
    arg2 = 2 * L * z / ((1 - rho**2) * h)
    pdf = aux * iv(0, arg1) * kv(L - 1, arg2)
    return pdf
def log_pdf_prod_intensities(z, L, rho, mu1, mu2):
    aux = pdf_prod_intensities(z, L, rho, mu1, mu2)
    log_pdf = np.log(aux)
    return log_pdf