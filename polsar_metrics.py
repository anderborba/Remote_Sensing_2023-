# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 16:39:53 2023

@author: ander
"""
import numpy as np
from scipy.spatial.distance import directed_hausdorff
#from hausdorff import hausdorff_distance
#
def hausdorff_distance_aab(GR, M, nrows, ncols):
    u = []
    for i in range(nrows):
        for j in range(ncols):
            if (GR[i, j] != 0):
                u.append([i,j])
    v = []
    for i in range(nrows):
        for j in range(ncols):
            if (M[i, j] != 0):
                v.append([i,j])
    dist1 = directed_hausdorff(u, v)[0]
    dist2 = directed_hausdorff(v, u)[0]
    distf = np.max((dist1, dist2))
    return distf
#def hausdorff_distance_th(GR, M):
#    dist1 = hausdorff_distance(GR, M, distance='euclidean')
#    dist2 = hausdorff_distance(GR, M, distance='euclidean')
#    distf = np.max((dist1, dist2))
#    return distf


#    list_grx = []
#    list_gry = []
#    for i in range(nrows):
#        for j in range(nrows):
#            if (GR[i,j] != 0):
#              list_grx.append(i)
#              list_gry.append(j)
#    dn = len(list_grx)
#    list_Mx = []
#    list_My = []
#    for i in range(nrows):
#        for j in range(nrows):
#            if (M[i,j] != 0):
#              list_Mx.append(i)
#              list_My.append(j)
#    dm = len(list_Mx)
#    #
#    p = np.zeros(2)
#    pa = np.zeros(2)
#    # Set the minor set to do max in hausdorff distance
#    #
#    print(dn, dm)
#    minor = np.min((dn, dm))
#    print(minor)
#    major  = np.max((dn, dm))
#    vet_min = np.zeros(major)
#    vet_max = np.zeros(minor)
#    for l in range(minor):
#        p[0] = list_grx[l]
#        p[1] = list_gry[l]
#        print(p[0])
#        print(p[1])
#        for i in range(major):
#            pa[0] = list_Mx[i]
#            pa[1] = list_My[i]
#            print(pa[0])
#            print(pa[1])
#            #vet_min[i] = la.norm(p-pa)
#            vet_min[i] = np.sqrt((p[0]-pa[0])**2 + (p[1]-pa[1])**2)
#        norm_min = np.min(vet_min)
#        vet_max[l] = norm_min
#    print(vet_max)
#    print(np.max(vet_max))
#    return
