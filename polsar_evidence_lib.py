## Version 03/08/2021
# Article GRSL
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
#import math
from scipy.optimize import dual_annealing
#### Used to find_evidence_bfgs
from scipy.optimize import minimize
import matplotlib.pyplot as plt
#
import polsar_basics as pb
import polsar_loglikelihood as plk
import polsar_total_loglikelihood as ptl
import polsar_plot as pplt
#
## Finds border evidences
def find_evidence(RAIO, NUM_RAIOS, canal, MY, lim):
    print("Computing evidence - this might take a while")
    z = np.zeros(RAIO)
    Le = 4
    Ld = 4
    evidencias = np.zeros(NUM_RAIOS)
    for k in range(NUM_RAIOS):
        z = MY[k, :, canal]
        zaux = np.zeros(RAIO)
        conta = 0
        for i in range(RAIO):
            if z[i] > 0:
                zaux[conta] = z[i]
                conta = conta + 1
        #
        indx  = pb.get_indexes(zaux != 0)
        N = int(np.max(indx))
        z =  zaux[1:N]
        matdf1 =  np.zeros((N, 2))
        matdf2 =  np.zeros((N, 2))
        for j in range(1, N):
            mue = sum(z[0: j]) / j
            matdf1[j, 0] = Le
            matdf1[j, 1] = mue
            mud = sum(z[j: (N + 1)]) / (N - j)
            matdf2[j, 0] = Ld
            matdf2[j, 1] = mud
        #
        lw = [lim]
        up = [N - lim]
        #
        #polplt.plot_total_likelihood(z, N, matdf1, matdf2)
        ret = dual_annealing(lambda x:ptl.func_obj_l_L_mu(x,z, N, matdf1, matdf2),
                                 bounds=list(zip(lw, up)),
                                 seed=1234)
        evidencias[k] = np.round(ret.x)
    return evidencias
#
##Finds border evidences using BFGS to estimate the parameters.
##Using: 1) MLE - Maximum Likelihood Estimation.
##    2) Optimization method BFGS to estimate the gamma pdf  parameters.
##    3) Optimization method Simulated annealing to detect edge border evidences.
#
def find_evidence_bfgs(RAIO, NUM_RAIOS, canal, MY, lim):
    print("Computing evidence with bfgs - this might take a while")
    z = np.zeros(RAIO)
    evidencias = np.zeros(NUM_RAIOS)
    # Put limit lower bound (lb) to variables
    # Put limit upper bound (ub) to variables
    lb = 0.00000001
    ub = 10
    bnds = ((lb, ub), (lb, ub))
    zeros_count = np.zeros(NUM_RAIOS)
    for k in range(NUM_RAIOS):
        z = MY[k, :, canal]
        zaux = np.zeros(RAIO)
        conta = 0
        for i in range(RAIO):
            if z[i] > 0:
                zaux[conta] = z[i]
                conta = conta + 1
        #
        indx  = pb.get_indexes(zaux != 0)
        N = int(np.max(indx)) + 1
        zeros_count[k] = N
        z =  zaux[0: N]
        matdf1 =  np.zeros((N, 2))
        matdf2 =  np.zeros((N, 2))
        varx = np.zeros(2)
        for j in range(1, N):
            varx[0] = 1
            varx[1] = sum(z[0: j]) / j
            res = minimize(lambda varx:plk.loglike(varx, z, j),
                                varx,
                                method='L-BFGS-B',
                                bounds= bnds)
            matdf1[j, 0] = res.x[0]
            matdf1[j, 1] = res.x[1]
            #
            varx[0] = 1
            varx[1] = sum(z[j: N]) / (N - j)
            res = minimize(lambda varx:plk.loglikd(varx, z, j, N),
                                varx,
                                method='L-BFGS-B',
                                bounds= bnds)
            matdf2[j, 0] = res.x[0]
            matdf2[j, 1] = res.x[1]
        #
        lw = [lim]
        up = [N - lim]
        ret = dual_annealing(lambda x:ptl.func_obj_l_L_mu(x,z, N, matdf1, matdf2),
                                 bounds=list(zip(lw, up)),
                                 seed=1234)
        evidencias[k] = np.round(ret.x)
    return evidencias, zeros_count
#
def find_evidence_bfgs_zeros_adaptive(RAIO, NUM_RAIOS, canal, MY, lim):
    print("Computing evidence with bfgs - this might take a while")
    z = np.zeros(RAIO)
    evidencias = np.zeros(NUM_RAIOS)
    # Put limit lower bound (lb) to variables
    # Put limit upper bound (ub) to variables
    lb = 0.00000001
    ub = 10
    bnds = ((lb, ub), (lb, ub))
    #
    zeros_count = np.zeros(NUM_RAIOS)
    zaux = []
    vetaux = []
    for k in range(NUM_RAIOS):
        aux = []
        for i in range(RAIO):
            if MY[k, i, canal] > 0:
                    aux.append(MY[k, i, canal])
        #
        zaux.append(aux)
        N = len(aux)
        zeros_count[k] = N
    #    th = 15
    #    if ((RAIO - zeros_count[k]) <= th):
        vetaux.append(k)
    #
    #
    for k in vetaux:
        N = int(zeros_count[k])
        z =  np.array(zaux[k])
        matdf1 =  np.zeros((N, 2))
        matdf2 =  np.zeros((N, 2))
        varx = np.zeros(2)
        for j in range(1, N):
            varx[0] = 1
            varx[1] = sum(z[0: j]) / j
            res = minimize(lambda varx:plk.loglike(varx, z, j),
                                varx,
                                method='L-BFGS-B',
                                bounds= bnds)
            matdf1[j, 0] = res.x[0]
            matdf1[j, 1] = res.x[1]
            #
            varx[0] = 1
            varx[1] = sum(z[j: N]) / (N - j)
            res = minimize(lambda varx:plk.loglikd(varx, z, j, N),
                                varx,
                                method='L-BFGS-B',
                                bounds= bnds)
            matdf2[j, 0] = res.x[0]
            matdf2[j, 1] = res.x[1]
        #
        lw = [lim]
        up = [N - lim]
        ret = dual_annealing(lambda x:ptl.func_obj_l_L_mu(x,z, N, matdf1, matdf2),
                                 bounds=list(zip(lw, up)),
                                 seed=1234)
        evidencias[k] = np.round(ret.x)
    return evidencias, zeros_count
#
def find_evidence_bfgs_zeros_adaptive_loglike_adaptive(RAIO, NUM_RAIOS, canal, MY, lim):
    print("Computing evidence with bfgs - this might take a while")
    z = np.zeros(RAIO)
    evidencias = np.zeros(NUM_RAIOS)
    # Put limit lower bound (lb) to variables
    # Put limit upper bound (ub) to variables
    lb = 0.00000001
    ub = 10
    bnds = ((lb, ub), (lb, ub))
    #
    zeros_count = np.zeros(NUM_RAIOS)
    zaux = []
    vetaux = []
    for k in range(NUM_RAIOS):
        aux = []
        for i in range(RAIO):
            if MY[k, i, canal] > 0:
                    aux.append(MY[k, i, canal])
        #
        zaux.append(aux)
        N = len(aux)
        zeros_count[k] = N
    #    th = 15
    #    if ((RAIO - zeros_count[k]) <= th):
        vetaux.append(k)
    #
    #
    for k in vetaux:
        N = int(zeros_count[k])
        z =  np.array(zaux[k])
        matdf1 =  np.zeros((N, 2))
        matdf2 =  np.zeros((N, 2))
        varx = np.zeros(2)
        for j in range(1, N):
            varx[0] = 1
            varx[1] = sum(z[0: j]) / j
            res = minimize(lambda varx:plk.loglike(varx, z, j),
                                varx,
                                method='L-BFGS-B',
                                bounds= bnds)
            matdf1[j, 0] = res.x[0]
            matdf1[j, 1] = res.x[1]
            #
            varx[0] = 1
            varx[1] = sum(z[j: N]) / (N - j)
            res = minimize(lambda varx:plk.loglikd(varx, z, j, N),
                                varx,
                                method='L-BFGS-B',
                                bounds= bnds)
            matdf2[j, 0] = res.x[0]
            matdf2[j, 1] = res.x[1]
        #
        lw = [lim]
        up = [N - lim]
        ret = dual_annealing(lambda x:ptl.func_obj_l_L_mu(x,z, N, matdf1, matdf2),
                                 bounds=list(zip(lw, up)),
                                 seed=1234)
        aux1 = ptl.func_obj_l_L_mu(lw             ,z, N, matdf1, matdf2)
        aux2 = ptl.func_obj_l_L_mu(np.round(ret.x),z, N, matdf1, matdf2)
        aux3 = ptl.func_obj_l_L_mu(up             ,z, N, matdf1, matdf2)
        #print(k)
        #print(aux1)
        #print(aux2)
        #print(aux3)
        aux_dif = np.abs((aux1 + aux2) * 0.5)
        value = np.abs(np.abs(aux2) - aux_dif)
        threshold = 2.92
        print("value")
        print(value)
        if value > threshold:
            evidencias[k] = np.round(ret.x)
        #evidencias[k] = np.round(ret.x)
    return evidencias, zeros_count
#
#
def find_evidence_bfgs_same_region_threshold(RAIO, NUM_RAIOS, canal, MY, lim):
    print("Computing evidence with bfgs - this might take a while - Region threshold")
    z = np.zeros(RAIO)
    evidencias = np.zeros(NUM_RAIOS)
    # Put limit lower bound (lb) to variables
    # Put limit upper bound (ub) to variables
    lb = 0.00000001
    ub = 10
    bnds = ((lb, ub), (lb, ub))
    zeros_count = np.zeros(NUM_RAIOS)
    for k in range(NUM_RAIOS):
    #for k in range(10,11):
        z = MY[k, :, canal]
        zaux = np.zeros(RAIO)
        conta = 0
        for i in range(RAIO):
            if z[i] > 0:
                zaux[conta] = z[i]
                conta = conta + 1
        #
        indx  = pb.get_indexes(zaux != 0)
        N = int(np.max(indx)) + 1
        zeros_count[k] = N
        z =  zaux[0: N]
        matdf1 =  np.zeros((N, 2))
        matdf2 =  np.zeros((N, 2))
        varx = np.zeros(2)
        for j in range(1, N):
            varx[0] = 1
            varx[1] = sum(z[0: j]) / j
            res = minimize(lambda varx:plk.loglike(varx, z, j),
                                varx,
                                method='L-BFGS-B',
                                bounds= bnds)
            matdf1[j, 0] = res.x[0]
            matdf1[j, 1] = res.x[1]
            #
            varx[0] = 1
            varx[1] = sum(z[j: N]) / (N - j)
            res = minimize(lambda varx:plk.loglikd(varx, z, j, N),
                                varx,
                                method='L-BFGS-B',
                                bounds= bnds)
            matdf2[j, 0] = res.x[0]
            matdf2[j, 1] = res.x[1]
        #
        lw = [lim]
        up = [N - lim]
        ret = dual_annealing(lambda x:ptl.func_obj_l_L_mu(x,z, N, matdf1, matdf2),
                                 bounds=list(zip(lw, up)),
                                 seed=1234)
        #
        value = ptl.func_obj_l_L_mu(ret.x, z, N, matdf1, matdf2)
        threshold_sup, threshold_inf = pb.same_region_threshold(z, N, matdf1, matdf2)
        if  ~(threshold_inf < value < threshold_sup):
            evidencias[k] = np.round(ret.x)
    return evidencias, zeros_count
#
def teste_limiar(RAIO, NUM_RAIOS, canal, MY, lim):
    print("Computing evidence with bfgs - this might take a while")
    z = np.zeros(RAIO)
    evidencias = np.zeros(NUM_RAIOS)
    # Put limit lower bound (lb) to variables
    # Put limit upper bound (ub) to variables
    lb = 0.00000001
    ub = 10
    bnds = ((lb, ub), (lb, ub))
    #
    zeros_count = np.zeros(NUM_RAIOS)
    zaux = []
    vetaux = []
    for k in range(NUM_RAIOS):
        aux = []
        for i in range(RAIO):
            if MY[k, i, canal] > 0:
                    aux.append(MY[k, i, canal])
        #
        zaux.append(aux)
        N = len(aux)
        zeros_count[k] = N
    #    th = 15
    #    if ((RAIO - zeros_count[k]) <= th):
        vetaux.append(k)
    #
    #
    vet_limiar = np.zeros(NUM_RAIOS)
    for k in vetaux:
        print(k)
        #N = int(zeros_count[k])
        N = 70
        z =  np.array(zaux[k])
        #matdf1 =  np.zeros((N, 2))
        #matdf2 =  np.zeros((N, 2))
        varx = np.zeros(2)
        #for j in range(1, N):
        varx[0] = 1
        varx[1] = sum(z[0: N]) / N
        res = minimize(lambda varx:plk.loglike(varx, z, N),
                                varx,
                                method='L-BFGS-B',
                                bounds= bnds)
        #matdf1[j, 0] = res.x[0]
        #matdf1[j, 1] = res.x[1]
        varx[0] = res.x[0]    #
        varx[1] = res.x[1]    #
        vet_limiar[k] = plk.loglike(varx, z, N)
    media = np.mean(vet_limiar)
    dp = np.std(vet_limiar)
    print("Vet l")
    print(vet_limiar)
    print(media)
    print(dp)
    limiar = media + 2 * dp
    print(limiar)
    return evidencias, zeros_count

##Finds border evidences using BFGS to estimate the parameters.
##Using: 1) MLE - Maximum Likelihood Estimation.
##    2) Optimization method BFGS to estimate the gamma pdf  parameters.
##    3) Optimization method Simulated annealing to detect edge border evidences.
##    4) Using PDF to span
#
def find_evidence_bfgs_span(RAIO, NUM_RAIOS, MY, lim):
    print("Computing evidence with bfgs to span PDF - this might take a while")
    z = np.zeros(RAIO)
    evidencias = np.zeros(NUM_RAIOS)
    lb = 0.00000001
    ub = 10
    bnds = ((lb, ub), (lb, ub))
    zeros_count = np.zeros(NUM_RAIOS)
    for k in range(NUM_RAIOS):
        zaux = np.zeros(RAIO)
        z = MY[k, :, 0] + 2 * MY[k, :, 1] + MY[k, :, 2]
        conta = 0
        for i in range(RAIO):
            if z[i] > 0:
                zaux[conta] = z[i]
                conta = conta + 1
        #
        indx  = pb.get_indexes(zaux != 0)
        N = int(np.max(indx)) + 1
        zeros_count[k] = N
        z =  zaux[0: N]
        matdf1 =  np.zeros((N, 2))
        matdf2 =  np.zeros((N, 2))
        varx = np.zeros(2)
        for j in range(1, N):
            varx[0] = 1
            varx[1] = sum(z[0: j]) / j
            res = minimize(lambda varx:plk.loglike(varx, z, j),
                            varx,
                            method='L-BFGS-B',
                            bounds= bnds)
            matdf1[j, 0] = res.x[0]
            matdf1[j, 1] = res.x[1]
            #
            varx[0] = 1
            varx[1] = sum(z[j: N]) / (N - j)
            res = minimize(lambda varx:plk.loglikd(varx, z, j, N),
                            varx,
                            method='L-BFGS-B',
                            bounds= bnds)
            matdf2[j, 0] = res.x[0]
            matdf2[j, 1] = res.x[1]
            #
            #
        lw = [lim]
        up = [N - lim]
        ret = dual_annealing(lambda x:ptl.func_obj_l_L_mu(x,z, N, matdf1, matdf2),
                              bounds=list(zip(lw, up)),
                              seed=1234)
        evidencias[k] = np.round(ret.x)
    return evidencias, zeros_count
##Finds border evidences using BFGS to estimate the parameters.
##Using: 1) MLE - Maximum Likelihood Estimation.
##    2) Optimization method BFGS to estimate the gamma pdf  parameters.
##    3) Optimization method Simulated annealing to detect edge border evidences.
##    4) Using PDF to span
#
def find_evidence_bfgs_span_same_region_threshold(RAIO, NUM_RAIOS, MY, lim):
    print("Computing evidence with bfgs to span PDF - this might take a while - same_region_threshold")
    z = np.zeros(RAIO)
    evidencias = np.zeros(NUM_RAIOS)
    lb = 0.00000001
    ub = 10
    bnds = ((lb, ub), (lb, ub))
    zeros_count = np.zeros(NUM_RAIOS)
    for k in range(NUM_RAIOS):
        zaux = np.zeros(RAIO)
        z = MY[k, :, 0] + 2 * MY[k, :, 1] + MY[k, :, 2]
        conta = 0
        for i in range(RAIO):
            if z[i] > 0:
                zaux[conta] = z[i]
                conta = conta + 1
        #
        indx  = pb.get_indexes(zaux != 0)
        N = int(np.max(indx)) + 1
        zeros_count[k] = N
        z =  zaux[0: N]
        matdf1 =  np.zeros((N, 2))
        matdf2 =  np.zeros((N, 2))
        varx = np.zeros(2)
        for j in range(1, N):
            varx[0] = 1
            varx[1] = sum(z[0: j]) / j
            res = minimize(lambda varx:plk.loglike(varx, z, j),
                            varx,
                            method='L-BFGS-B',
                            bounds= bnds)
            matdf1[j, 0] = res.x[0]
            matdf1[j, 1] = res.x[1]
            #
            varx[0] = 1
            varx[1] = sum(z[j: N]) / (N - j)
            res = minimize(lambda varx:plk.loglikd(varx, z, j, N),
                            varx,
                            method='L-BFGS-B',
                            bounds= bnds)
            matdf2[j, 0] = res.x[0]
            matdf2[j, 1] = res.x[1]
            #
            #
        lw = [lim]
        up = [N - lim]
        ret = dual_annealing(lambda x:ptl.func_obj_l_L_mu(x,z, N, matdf1, matdf2),
                              bounds=list(zip(lw, up)),
                              seed=1234)
        #
        value = ptl.func_obj_l_L_mu(ret.x, z, N, matdf1, matdf2)
        threshold_sup, threshold_inf = pb.same_region_threshold(z, N, matdf1, matdf2)
        if  ~(threshold_inf < value < threshold_sup):
            evidencias[k] = np.round(ret.x)
    return evidencias, zeros_count
#
##Finds border evidences using BFGS to estimate the parameters in
## intensity ratio distribution.
##Using: 1) MLE - Maximum Likelihood Estimation.
##    2) Optimization method BFGS to estimate the intensity ratio pdf  parameters.
##    3) Optimization method Simulated annealing to detect edge border evidences.
##    4) Using PDF to intensity ratio
#
def find_evidence_bfgs_intensity_ratio(RAIO, NUM_RAIOS, MY, lim, inum, idem):
    print("Computing evidence with bfgs to intensity ratio pdf - this might take a while")
    z = np.zeros(RAIO)
    evidencias = np.zeros(NUM_RAIOS)
    #
    lbtau = 0.00000001
    ubtau = 100
    lbrho = -0.99
    ubrho =  0.99
    bnds = ((lbtau, ubtau), (lbrho, ubrho))
    # Set L = 4 fixed
    L = 4
    for k in range(NUM_RAIOS):
        zaux = np.zeros(RAIO)
        conta = 0
        for i in range(RAIO):
            if MY[k, i, inum] > 0 and MY[k, i, idem] > 0:
                    zaux[conta] = MY[k, i, inum] / MY[k, i, idem]
                    conta = conta + 1
        #
        indx  = pb.get_indexes(zaux != 0)
        N = int(np.max(indx))
        z =  zaux[0: N]
        matdf1 =  np.zeros((N, 2))
        matdf2 =  np.zeros((N, 2))
        varx = np.zeros(2)
        for j in range(1, N):
            varx[0] = 1
            varx[1] = 0.1
            Ni = 0
            Nf = j
            res = minimize(lambda varx:plk.loglik_intensity_ratio(varx, z, Ni, Nf, L),
                                    varx,
                                    method='L-BFGS-B',
                                    bounds= bnds)
            #
            matdf1[j, 0] = res.x[0]
            matdf1[j, 1] = res.x[1]
            #
            varx[0] = 1
            varx[1] = 0.1
            Ni = j
            Nf = N - 1
            res = minimize(lambda varx:plk.loglik_intensity_ratio(varx, z, Ni, Nf, L),
                                    varx,
                                    method='L-BFGS-B',
                                    bounds= bnds)
            #
            matdf2[j, 0] = res.x[0]
            matdf2[j, 1] = res.x[1]
            #

        lw = [lim]
        up = [N - lim]
        ret = dual_annealing(lambda x:ptl.func_obj_l_intensity_ratio_tau_rho(x, z, N, matdf1, matdf2, L),
                                bounds=list(zip(lw, up)),
                                seed=1234)
        evidencias[k] = np.round(ret.x)
    return evidencias
##Finds border evidences using BFGS to estimate the parameters in
## intensity ratio distribution using three parameters.
##Using: 1) MLE - Maximum Likelihood Estimation.
##    2) Optimization method BFGS to estimate the intensity ratio pdf  parameters.
##    3) Optimization method Simulated annealing to detect edge border evidences.
##    4) Using PDF to intensity ratio
#
def find_evidence_bfgs_intensity_ratio_three_param(RAIO, NUM_RAIOS, MY, lim, inum, idem):
    print("Computing evidence with bfgs to intensity ratio pdf - this might take a while")
    ### Verificar se não é melhor colocar zaux
    z = np.zeros(RAIO)
    evidencias = np.zeros(NUM_RAIOS)
    #
    lbtau = 0.00000001
    ubtau = 10
    lbrho = -0.99
    ubrho =  0.99
    Ll    = 0.1
    Lu    = 10
    bnds = ((lbtau, ubtau), (lbrho, ubrho), (Ll, Lu))
    zeros_count = np.zeros(NUM_RAIOS)
    zaux = []
    z1 = []
    z2 = []
    est_tau = []
    est_rho = []
    est_L = []
    vetaux = []
    for k in range(NUM_RAIOS):
        aux = []
        for i in range(RAIO):
            if MY[k, i, inum] > 0 and MY[k, i, idem] > 0:
                    aux.append(MY[k, i, inum] / MY[k, i, idem])
                    z1.append(MY[k, i, inum])
                    z2.append(MY[k, i, idem])
        #
        est_tau.append(pb.initial_guest_tau(z1, z2))
        est_rho.append(pb.initial_guest_rho(z1, z2))
        est_L.append(pb.initial_guest_L(z1, z2))
        zaux.append(aux)
        N = len(aux)
        zeros_count[k] = N
        vetaux.append(k)
    #
    for k in vetaux:
        N = int(zeros_count[k])
        z =  np.array(zaux[k])
        matdf1 =  np.zeros((N, 3))
        matdf2 =  np.zeros((N, 3))
        varx = np.zeros(3)
        for j in range(1, N):
            varx[0] = est_tau[k]
            varx[1] = est_rho[k]
            varx[2] = est_L[k]
            #varx[0] = 1
            #varx[1] = 0.1
            #varx[2] = 2
            Ni = 0
            Nf = j
            res = minimize(lambda varx:plk.loglik_intensity_ratio_three_param(varx, z, Ni, Nf),
                                    varx,
                                    method='L-BFGS-B',
                                    bounds= bnds)
            #
            matdf1[j, 0] = res.x[0]
            matdf1[j, 1] = res.x[1]
            matdf1[j, 2] = res.x[2]
            #
            #varx[0] = 1
            #varx[1] = 0.1
            #varx[2] = 2
            varx[0] = est_tau[k]
            varx[1] = est_rho[k]
            varx[2] = est_L[k]
            Ni = j
            Nf = N - 1
            res = minimize(lambda varx:plk.loglik_intensity_ratio_three_param(varx, z, Ni, Nf),
                                    varx,
                                    method='L-BFGS-B',
                                    bounds= bnds)
            #
            matdf2[j, 0] = res.x[0]
            matdf2[j, 1] = res.x[1]
            matdf2[j, 2] = res.x[2]
            #

        lw = [lim]
        up = [N - lim]
        ret = dual_annealing(lambda x:ptl.func_obj_l_intensity_ratio_tau_rho_L(x, z, N, matdf1, matdf2),
                                bounds=list(zip(lw, up)),
                                seed=1234)
        evidencias[k] = np.round(ret.x)
    return evidencias, zeros_count
##Finds border evidences using BFGS to estimate the parameters in
## intensity ratio distribution using three parameters.
##Using: 1) MLE - Maximum Likelihood Estimation.
##    2) Optimization method BFGS to estimate the intensity ratio pdf  parameters.
##    3) Optimization method Simulated annealing to detect edge border evidences.
##    4) Using PDF to intensity ratio
#
def find_evidence_bfgs_intensity_ratio_three_param_same_region_threshold(RAIO, NUM_RAIOS, MY, lim, inum, idem):
    print("Computing evidence with bfgs to intensity ratio pdf - this might take a while - Same region threshold")
    ### Verificar se não é melhor colocar zaux
    z = np.zeros(RAIO)
    evidencias = np.zeros(NUM_RAIOS)
    #
    lbtau = 0.00000001
    ubtau = 10
    lbrho = -0.99
    ubrho =  0.99
    Ll    = 0.1
    Lu    = 10
    bnds = ((lbtau, ubtau), (lbrho, ubrho), (Ll, Lu))
    zeros_count = np.zeros(NUM_RAIOS)
    zaux = []
    z1 = []
    z2 = []
    est_tau = []
    est_rho = []
    est_L = []
    vetaux = []
    for k in range(NUM_RAIOS):
        aux = []
        for i in range(RAIO):
            if MY[k, i, inum] > 0 and MY[k, i, idem] > 0:
                    aux.append(MY[k, i, inum] / MY[k, i, idem])
                    z1.append(MY[k, i, inum])
                    z2.append(MY[k, i, idem])
        #
        est_tau.append(pb.initial_guest_tau(z1, z2))
        est_rho.append(pb.initial_guest_rho(z1, z2))
        est_L.append(pb.initial_guest_L(z1, z2))
        zaux.append(aux)
        N = len(aux)
        zeros_count[k] = N
        vetaux.append(k)
    #
    for k in vetaux:
        N = int(zeros_count[k])
        z =  np.array(zaux[k])
        matdf1 =  np.zeros((N, 3))
        matdf2 =  np.zeros((N, 3))
        varx = np.zeros(3)
        for j in range(1, N):
            varx[0] = est_tau[k]
            varx[1] = est_rho[k]
            varx[2] = est_L[k]
            #varx[0] = 1
            #varx[1] = 0.1
            #varx[2] = 2
            Ni = 0
            Nf = j
            res = minimize(lambda varx:plk.loglik_intensity_ratio_three_param(varx, z, Ni, Nf),
                                    varx,
                                    method='L-BFGS-B',
                                    bounds= bnds)
            #
            matdf1[j, 0] = res.x[0]
            matdf1[j, 1] = res.x[1]
            matdf1[j, 2] = res.x[2]
            #
            #varx[0] = 1
            #varx[1] = 0.1
            #varx[2] = 2
            varx[0] = est_tau[k]
            varx[1] = est_rho[k]
            varx[2] = est_L[k]
            Ni = j
            Nf = N - 1
            res = minimize(lambda varx:plk.loglik_intensity_ratio_three_param(varx, z, Ni, Nf),
                                    varx,
                                    method='L-BFGS-B',
                                    bounds= bnds)
            #
            matdf2[j, 0] = res.x[0]
            matdf2[j, 1] = res.x[1]
            matdf2[j, 2] = res.x[2]
            #

        lw = [lim]
        up = [N - lim]
        ret = dual_annealing(lambda x:ptl.func_obj_l_intensity_ratio_tau_rho_L(x, z, N, matdf1, matdf2),
                                bounds=list(zip(lw, up)),
                                seed=1234)
        value = ptl.func_obj_l_intensity_ratio_tau_rho_L(ret.x, z, N, matdf1, matdf2)
        threshold_sup, threshold_inf = pb.same_region_threshold(z, N, matdf1, matdf2)
        if  ~(threshold_inf < value < threshold_sup):
            evidencias[k] = np.round(ret.x)
    return evidencias, zeros_count
##Finds border evidences using BFGS to estimate the parameters in
## intensity ratio distribution using three parameters.
##Using: 1) MLE - Maximum Likelihood Estimation.
##    2) Optimization method BFGS to estimate the intensity ratio pdf  parameters.
##    3) Optimization method Simulated annealing to detect edge border evidences.
##    4) Using PDF to intensity ratio
#
def find_evidence_bfgs_intensity_ratio_three_param_threshold(RAIO, NUM_RAIOS, MY, lim, inum, idem):
    print("Computing evidence with bfgs to intensity ratio pdf - this might take a while")
    ### Verificar se não é melhor colocar zaux
    z = np.zeros(RAIO)
    evidencias = np.zeros(NUM_RAIOS)
    #
    lbtau = 0.00000001
    ubtau = 10
    lbrho = -0.99
    ubrho =  0.99
    Ll    = 0.1
    Lu    = 10
    bnds = ((lbtau, ubtau), (lbrho, ubrho), (Ll, Lu))
    zeros_count = np.zeros(NUM_RAIOS)
    zaux = []
    vetaux = []
    for k in range(NUM_RAIOS):
        aux = []
        for i in range(RAIO):
            if MY[k, i, inum] > 0 and MY[k, i, idem] > 0:
                    aux.append(MY[k, i, inum] / MY[k, i, idem])
        #
        zaux.append(aux)
        N = len(aux)
        zeros_count[k] = N
        th = 50
        if ((RAIO - n_zeros_count[k]) <= th):
            vetaux.append(k)
    #
    for k in vetaux:
        N = int(zeros_count[k])
        z =  np.array(zaux[k])
        matdf1 =  np.zeros((N, 3))
        matdf2 =  np.zeros((N, 3))
        varx = np.zeros(3)
        for j in range(1, N):
            varx[0] = 1
            varx[1] = 0.1
            varx[2] = 2
            Ni = 0
            Nf = j
            res = minimize(lambda varx:plk.loglik_intensity_ratio_three_param(varx, z, Ni, Nf),
                                    varx,
                                    method='L-BFGS-B',
                                    bounds= bnds)
            #
            matdf1[j, 0] = res.x[0]
            matdf1[j, 1] = res.x[1]
            matdf1[j, 2] = res.x[2]
            #
            varx[0] = 1
            varx[1] = 0.1
            varx[2] = 2
            Ni = j
            Nf = N - 1
            res = minimize(lambda varx:plk.loglik_intensity_ratio_three_param(varx, z, Ni, Nf),
                                    varx,
                                    method='L-BFGS-B',
                                    bounds= bnds)
            #
            matdf2[j, 0] = res.x[0]
            matdf2[j, 1] = res.x[1]
            matdf2[j, 2] = res.x[2]
            #

        lw = [lim]
        up = [N - lim]
        ret = dual_annealing(lambda x:ptl.func_obj_l_intensity_ratio_tau_rho_L(x, z, N, matdf1, matdf2),
                                bounds=list(zip(lw, up)),
                                seed=1234)
        evidencias[k] = np.round(ret.x)
    return evidencias, zeros_count
##Finds border evidences using BFGS to estimate the parameters in
## magnitude of product of intensity distribution.
##Using: 1) MLE - Maximum Likelihood Estimation.
##    2) Optimization method BFGS to estimate the pdf product of intensity  parameters.
##    3) Optimization method Simulated annealing to detect edge border evidences.
##    4) Using PDF to the product of intensity
#
def find_evidence_bfgs_prod_int(RAIO, NUM_RAIOS, MY, lim, mul1, mul2):
    print("Computing evidence with bfgs to product intensities pdf - this might take a while")
    z = np.zeros(RAIO)
    evidencias = np.zeros(NUM_RAIOS)
#
    lb = 0.0001
    ub = 10
    lbrho = 0.00001
    ubrho =  0.999999
    lbmu1 = 0.00001
    ubmu1 = 10
    lbmu2 = 0.00001
    ubmu2 = 10
    bnds = ((lb, ub), (lbrho, ubrho), (lbmu1, ubmu1), (lbmu2, ubmu2))
    N = RAIO
    n_zeros_count = np.zeros(NUM_RAIOS)
    for k in range(NUM_RAIOS):
        zaux = np.zeros(RAIO)
        for i in range(RAIO):
            zaux[i] = MY[k, i, mul1] * MY[k, i, mul2]
        #
        indx  = pb.get_indexes(zaux != 0)
        N = int(np.max(indx))
        n_zeros_count[k] = N
        z =  zaux[0: N]
        #
        matdf1 =  np.zeros((N, 4))
        matdf2 =  np.zeros((N, 4))
        varx = np.zeros(4)
        for j in range(1, N):
            varx[0] = 2
            varx[1] = 0.5
            varx[2] = sum(z[0: j]) / j
            varx[3] = sum(z[0: j]) / j
            Ni = 0
            Nf = j
            res = minimize(lambda varx:plk.loglik_intensity_prod(varx, z, Ni, Nf),
                                    varx,
                                    method='L-BFGS-B',
                                    bounds= bnds)
            matdf1[j, 0] = res.x[0]
            matdf1[j, 1] = res.x[1]
            matdf1[j, 2] = res.x[2]
            matdf1[j, 3] = res.x[3]
            #
            varx[0] = 2
            varx[1] = 0.5
            varx[2] = sum(z[j: N]) / (N - j)
            varx[3] = sum(z[j: N]) / (N - j)
            Ni = j
            Nf = N
            res = minimize(lambda varx:plk.loglik_intensity_prod(varx, z, Ni, Nf),
                                    varx,
                                    method='L-BFGS-B',
                                    bounds= bnds)
            matdf2[j, 0] = res.x[0]
            matdf2[j, 1] = res.x[1]
            matdf2[j, 2] = res.x[2]
            matdf2[j, 3] = res.x[3]
        #
        lw = [lim]
        up = [N - lim]
        #pplt.plot_log_pdf_prod_intensities(matdf2[k, 0], matdf2[k, 1],matdf2[k, 2], matdf2[k, 3])
        #pplt.plot_pdf_prod_intensities(4, 0.9, 0.03, 0.05)
        #pplt.plot_total_likelihood_prod_int(z, N, matdf1, matdf2)
        #pplt.plot_pdf_prod_intensities(z, matdf1[j, 0], matdf1[j, 1],matdf1[j, 2], matdf1[j, 3])
        #pplt.plot_log_pdf_prod_intensities(z, matdf2[k, 0], matdf2[k, 1],matdf2[k, 2], matdf2[k, 3])
        #pplt.plot_total_likelihood_prod_int_sum(z, N, matdf1, matdf2)
        ret = dual_annealing(lambda x:ptl.func_obj_l_intensity_prod(x, z, N, matdf1, matdf2),
                                bounds=list(zip(lw, up)),
                                seed=1234)
        evidencias[k] = np.round(ret.x)
    print("n_zeros_count")
    print(n_zeros_count)
    return evidencias
#
##Finds border evidences using BFGS to estimate the parameters with
## bivariate distribution using four parameters.
##Using: 1) MLE - Maximum Likelihood Estimation.
##    2) Optimization method BFGS to estimate the bivariate pdf  parameters.
##    3) Optimization method Simulated annealing to detect edge border evidences.
##    4) Using PDF to the a PDF product of the intensities bivariate
#
def find_evidence_bfgs_int_prod_biv(RAIO, NUM_RAIOS, MY, lim, c1, c2):
    print("Computing evidence with bfgs to intensity ratio pdf - this might take a while")
    ### Verificar se não é melhor colocar zaux
    z1 = np.zeros(RAIO)
    z2 = np.zeros(RAIO)
    evidencias = np.zeros(NUM_RAIOS)
    #
    lbrho = -0.99
    ubrho =  0.99
    Ll    = 0.1
    Lu    = 10
    lbs1  = 0.00000001
    ubs1  = 1
    lbs2  = 0.00000001
    ubs2  = 1
    bnds = ((lbrho, ubrho), (Ll, Lu), (lbs1, ubs1), (lbs2, ubs2))
    #for k in range(NUM_RAIOS):
    for k in range(79,80):
        zaux1 = np.zeros(RAIO)
        zaux2 = np.zeros(RAIO)
        conta1 = 0
        conta2 = 0
        for i in range(RAIO):
            if MY[k, i, c1] > 0:
                zaux1[conta1] = MY[k, i, c1]
                conta1 = conta1 + 1
            if MY[k, i, c2] > 0:
                zaux2[conta2] = MY[k, i, c2]
                conta2 = conta2 + 1

        #        Fazer o ratio!!!!!!!!!!!!!!!!
        #
        indx1  = pb.get_indexes(zaux1 != 0)
        indx2  = pb.get_indexes(zaux2 != 0)
        N1 = int(np.max(indx1)) + 1
        N2 = int(np.max(indx2)) + 1
        N = np.min([N1, N2])
        z1[0: N] =  zaux1[0: N]
        z2[0: N] =  zaux2[0: N]
        matdf1 =  np.zeros((N, 4))
        matdf2 =  np.zeros((N, 4))
        varx = np.zeros(4)
        res = np.zeros(4)
        #for j in range(1, N):
        for j in range(10, 11):
            varx[0] = 0.1
            varx[1] = 2
            varx[2] = sum(z1[0: j]) / j
            varx[3] = sum(z2[0: j]) / j
            pplt.plot_pdf_prod_intensities_biv(varx[0], varx[1], varx[2], varx[3])
            Ni = 0
            Nf = j
            #res = minimize(lambda varx:plk.loglik_intensity_prod_biv(varx, z1, z2, Ni, Nf),
            #                        varx,
            #                        method='L-BFGS-B',
            #                        bounds= bnds)
            #
            #matdf1[j, 0] = res.x[0]
            #matdf1[j, 1] = res.x[1]
            #matdf1[j, 2] = res.x[2]
            #matdf1[j, 3] = res.x[3]
            #
            varx[0] = 0.1
            varx[1] = 2
            varx[2] = sum(z1[j: (N + 1)]) / (N - j)
            varx[3] = sum(z2[j: (N + 1)]) / (N - j)
            Ni = j
            Nf = N - 1
            res = 0
            #res = minimize(lambda varx:plk.loglik_intensity_prod_biv(varx, z1, z2, Ni, Nf),
            #                        varx,
            #                        method='L-BFGS-B',
            #                        bounds= bnds)
            #
            #matdf2[j, 0] = res.x[0]
            #matdf2[j, 1] = res.x[1]
            #matdf2[j, 2] = res.x[2]
            #matdf2[j, 3] = res.x[3]
            #
        #pplt.plot_total_likelihood_prod_biv(z1, z2, N, matdf1, matdf2)
        lw = [lim]
        up = [N - lim]
        ret = 0
        #ret = dual_annealing(lambda x:ptl.func_obj_l_intensity_prod_biv(x, z1, z2, N, matdf1, matdf2),
        #                        bounds=list(zip(lw, up)),
        #                        seed=1234)
        #evidencias[k] = np.round(ret.x)
        evidencias[k] = 0
    return evidencias
# Set evidence in an image
def add_evidence(nrows, ncols, ncanal, evidencias, NUM_RAIOS, MXC, MYC):
    IM  = np.zeros([nrows, ncols, ncanal])
    for canal in range(ncanal):
        for k in range(NUM_RAIOS):
            ik = int(evidencias[k, canal])
            ia = int(MXC[k, ik])
            ja = int(MYC[k, ik])
            IM[ja, ia, canal] = 1
    return IM
# Set evidence in an image
def add_evidence_without_zeros(nrows, ncols, ncanal, evidencias, NUM_RAIOS, MXC, MYC):
    IM  = np.zeros([nrows, ncols, ncanal])
    for canal in range(ncanal):
        for k in range(NUM_RAIOS):
            if evidencias[k, canal] != 0:
                ik = int(evidencias[k, canal])
                ia = int(MXC[k, ik])
                ja = int(MYC[k, ik])
                IM[ja, ia, canal] = 1
    return IM
#
## Shows the evidence
#def show_evidence(pauli, NUM_RAIOS, MXC, MYC, img_rt, evidence, banda):
#	PIA=pauli.copy()
#	plt.figure(figsize=(20*img_rt, 20))
#	for k in range(NUM_RAIOS):
#    		ik = np.int(evidence[k, banda])
#    		ia = np.int(MXC[k, ik])
#    		ja = np.int(MYC[k, ik])
#    		plt.plot(ia, ja, marker='o', color="darkorange")
#	plt.imshow(PIA)
#	plt.show()
# Set evidence in a simulated image
#def add_evidence_simulated(nrows, ncols, ncanal, evidencias):
#    IM  = np.zeros([nrows, ncols, ncanal])
#    for canal in range(ncanal):
#        for k in range(nrows):
#            ik = np.int(evidencias[k, canal])
#            IM[ik, k, canal] = 1
#    return IM
## Shows the evidence in simulated image
#def show_evidence_simulated(pauli, NUM_RAIOS, img_rt, evidence, banda):
#	PIA=pauli.copy()
#	plt.figure(figsize=(20*img_rt, 20))
#	for k in range(NUM_RAIOS):
#    		ik = np.int(evidence[k, banda])
#    		plt.plot(ik, k, marker='o', color="darkorange")
#	plt.imshow(PIA)
#	plt.show()
